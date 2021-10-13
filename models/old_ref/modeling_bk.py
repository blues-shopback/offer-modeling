import numpy as np
import tensorflow as tf


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.

    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


class Embedding(tf.Module):

    def __init__(self, n_token, d_embed, initializer, name=None, dtype=tf.float32):
        super().__init__(name=name)
        self.n_token = n_token
        self.d_embed = d_embed
        self.initializer = initializer
        self.dtype = dtype

    @tf.Module.with_name_scope
    def __call__(self, x):
        if not hasattr(self, "lookup_table"):
            self.lookup_table = tf.Variable(
                self.initializer([self.n_token, self.d_embed], dtype=self.dtype))

        return tf.nn.embedding_lookup(self.lookup_table, x)


def positional_embedding(pos_seq, inv_freq, bsz=None):
    sinusoid_inp = tf.einsum('i,d->id', pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    pos_emb = tf.expand_dims(pos_emb, axis=1)
    # pos_emb = pos_emb[:, None, :]

    if bsz is not None:
        pos_emb = tf.tile(pos_emb, [1, bsz, 1])

    return pos_emb


def positional_encoding(qlen, d_model, clamp_len, bsz=None, dtype=tf.float32):
    """create relative positional encoding."""
    d_model_range = tf.constant(d_model, dtype=tf.int32)
    freq_seq = tf.range(
        tf.constant(0, dtype=tf.int32),
        d_model_range,
        tf.constant(2, dtype=tf.int32))

    freq_seq = tf.cast(freq_seq, dtype=dtype)
    inv_freq = 1 / (10000 ** (freq_seq / d_model))

    beg, end = qlen - 1, -1

    fwd_pos_seq = tf.range(
        beg,
        end,
        tf.constant(-1, dtype=tf.int32))
    if clamp_len > 0:
        fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -clamp_len, clamp_len)
    fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
    pos_emb = positional_embedding(fwd_pos_seq, inv_freq, bsz)

    return pos_emb


class PositionwiseFFN(tf.Module):

    def __init__(self, d_model, d_inner, dropout, initializer, activation_type='relu', name='ff',
                 is_training=True, pre_ln=True):
        super().__init__(name=name)
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.initializer = initializer
        self.activation_type = activation_type
        self.is_training = is_training
        self.pre_ln = pre_ln
        if self.activation_type == 'relu':
            self.act = tf.nn.relu
        elif self.activation_type == 'gelu':
            self.act = gelu
        else:
            raise ValueError('Unsupported activation type {}'.format(self.activation_type))

    @tf.Module.with_name_scope
    def __call__(self, inp):
        """Position-wise Feed-forward Network."""
        output = inp
        if self.pre_ln:
            if not hasattr(self, "layer_norm"):
                self.layer_norm = tf.keras.layers.LayerNormalization(name='LayerNorm')
            output = self.layer_norm(output)

        if not hasattr(self, "dense_1"):
            self.dense_1 = tf.keras.layers.Dense(
                self.d_inner, activation=self.act, kernel_initializer=self.initializer,
                name="layer_1")
            self.dense_2 = tf.keras.layers.Dense(
                self.d_model, activation=self.act, kernel_initializer=self.initializer,
                name="layer_2")
            self.dropout_layer = tf.keras.layers.Dropout(self.dropout, name="dropout")

        output = self.dense_1(output)
        output = self.dense_2(output)
        output = self.dropout_layer(output, training=self.is_training)

        output = output + inp
        if not self.pre_ln:
            if not hasattr(self, "layer_norm"):
                self.layer_norm = tf.keras.layers.LayerNormalization(name='LayerNorm')
            output = self.layer_norm(output)

        return output


class HeadProjection(tf.Module):
    def __init__(self, d_model, n_head, d_head, initializer, name=None, dtype=tf.float32):
        super().__init__(name=name)
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.initializer = initializer
        self.dtype = dtype

    @tf.Module.with_name_scope
    def __call__(self, inp):
        if not hasattr(self, "kernel"):
            self.kernel = tf.Variable(
                self.initializer([self.d_model, self.n_head, self.d_head], dtype=self.dtype),
                name="kernel"
            )
        head = tf.einsum('ibh,hnd->ibnd', inp, self.kernel)

        return head


class PostAttention(tf.Module):
    def __init__(self, d_model, n_head, d_head, dropout, initializer,
                 is_training=True, residual=True, pre_ln=True, name=None, dtype=tf.float32):
        super().__init__(name=name)
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.dropout = dropout
        self.initializer = initializer
        self.is_training = is_training
        self.residual = residual
        self.pre_ln = pre_ln
        self.dtype = dtype

    @tf.Module.with_name_scope
    def __call__(self, inp, attn_vec):
        if not hasattr(self, "kernel"):
            self.kernel = tf.Variable(
                self.initializer([self.d_model, self.n_head, self.d_head], dtype=self.dtype),
                name="proj/kernel")
        if not hasattr(self, "dropout_layer"):
            self.dropout_layer = tf.keras.layers.Dropout(self.dropout, name="dropout")
        attn_out = tf.einsum('ibnd,hnd->ibh', attn_vec, self.kernel)
        attn_out = self.dropout_layer(attn_out, training=self.is_training)
        if self.residual:
            attn_out = attn_out + inp

        if not self.pre_ln:
            if not hasattr(self, "layer_norm"):
                self.layer_norm = tf.keras.layers.LayerNormalization(name='LayerNorm')
            attn_out = self.layer_norm(attn_out)

        return attn_out


def abs_attn_core(q_head, k_head, v_head, attn_mask, dropatt, is_training,
                  scale):
    """Core absolute positional attention operations."""

    attn_score = tf.einsum('ibnd,jbnd->ijbn', q_head, k_head)
    attn_score *= scale
    if attn_mask is not None:
        attn_score = attn_score - 1e30 * attn_mask

    # attention probability
    attn_prob = tf.nn.softmax(attn_score, 1)
    drop_layer = tf.keras.layers.Dropout(dropatt)
    attn_prob = drop_layer(attn_prob, training=is_training)

    # attention output
    attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, v_head)

    return attn_vec


def rel_attn_core(q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                  r_w_bias, r_r_bias, r_s_bias, attn_mask, dropatt, is_training,
                  scale):
    """Core relative positional attention operations."""

    # content based attention score
    ac = tf.einsum('ibnd,jbnd->ijbn', q_head + r_w_bias, k_head_h)

    # position based attention score
    bd = tf.einsum('ibnd,jbnd->ijbn', q_head + r_r_bias, k_head_r)
    bd = rel_shift(bd, klen=tf.shape(ac)[1])

    # segment based attention score
    if seg_mat is None:
        ef = 0
    else:
        ef = tf.einsum('ibnd,snd->ibns', q_head + r_s_bias, seg_embed)
        ef = tf.einsum('ijbs,ibns->ijbn', seg_mat, ef)

    # merge attention scores and perform masking
    attn_score = (ac + bd + ef) * scale
    if attn_mask is not None:
        # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
        attn_score = attn_score - 1e30 * attn_mask

    # attention probability
    attn_prob = tf.nn.softmax(attn_score, 1)
    attn_prob = tf.nn.dropout(attn_prob, dropatt, training=is_training)

    # attention output
    attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)

    return attn_vec


def rel_shift(x, klen=-1):
    """perform relative shift to form the relative attention score."""
    x_size = tf.shape(x)

    x = tf.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
    x = tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])

    return x


def _create_mask(qlen, mlen, dtype=tf.float32, same_length=False):
    """create causal attention mask."""
    attn_mask = tf.ones([qlen, qlen], dtype=dtype)
    mask_u = tf.linalg.band_part(attn_mask, 0, -1)
    mask_dia = tf.linalg.band_part(attn_mask, 0, 0)
    attn_mask_pad = tf.zeros([qlen, mlen], dtype=dtype)
    ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
    if same_length:
        mask_l = tf.linalg.band_part(attn_mask, -1, 0)
        ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)

    return ret


def attention_mask(nd, ns, dtype):
    """Alternative function for _create_mask.
    1's in the lower triangle, counting from the lower right corner.
    """
    if isinstance(nd, int):
        nd = tf.constant(nd, dtype=tf.int32)
    if isinstance(ns, int):
        ns = tf.constant(ns, dtype=tf.int32)
    # i = tf.range(nd)[:, None]
    i = tf.expand_dims(tf.range(nd), axis=1)
    j = tf.range(ns)
    m = i <= j - ns + nd - 1
    return tf.cast(m, dtype)


def attention_mask_ops(nd, ns, dtype):
    """Alternative function for _create_mask.
    1's in the lower triangle, counting from the lower right corner.
    """
    if isinstance(nd, int):
        nd = tf.constant(nd, dtype=tf.int32)
    if isinstance(ns, int):
        ns = tf.constant(ns, dtype=tf.int32)
    # i = tf.range(nd)[:, None]
    i = tf.expand_dims(tf.range(nd), axis=1)
    # j = tf.range(ns)
    j = tf.range(ns-1, limit=-1, delta=-1)
    m = i > j - ns + nd
    return tf.cast(m, dtype)


def _cache_mem(curr_out, prev_mem, mem_len, reuse_len=None):
    """cache hidden states into memory."""
    if mem_len is None or mem_len == 0:
        return None
    else:
        if reuse_len is not None and reuse_len > 0:
            curr_out = curr_out[:reuse_len]

        if prev_mem is None:
            new_mem = curr_out[-mem_len:]
        else:
            new_mem = tf.concat([prev_mem, curr_out], 0)[-mem_len:]

    return tf.stop_gradient(new_mem)


def relative_positional_encoding(qlen, klen, d_model, clamp_len, attn_type,
                                 bi_data, bsz=None, dtype=None):
    """create relative positional encoding."""
    freq_seq = tf.range(0, d_model, 2.0)
    if dtype is not None and dtype != tf.float32:
        freq_seq = tf.cast(freq_seq, dtype=dtype)
    inv_freq = 1 / (10000 ** (freq_seq / d_model))

    if attn_type == 'bi':
        # beg, end = klen - 1, -qlen
        beg, end = klen, -qlen
    elif attn_type == 'uni':
        # beg, end = klen - 1, -1
        beg, end = klen, -1
    else:
        raise ValueError('Unknown `attn_type` {}.'.format(attn_type))

    if bi_data:
        fwd_pos_seq = tf.range(beg, end, -1.0)
        bwd_pos_seq = tf.range(-beg, -end, 1.0)

        if dtype is not None and dtype != tf.float32:
            fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
            bwd_pos_seq = tf.cast(bwd_pos_seq, dtype=dtype)

        if clamp_len > 0:
            fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -clamp_len, clamp_len)
            bwd_pos_seq = tf.clip_by_value(bwd_pos_seq, -clamp_len, clamp_len)

        if bsz is not None:
            # With bi_data, the batch size should be divisible by 2.
            assert bsz % 2 == 0
            fwd_pos_emb = positional_embedding(fwd_pos_seq, inv_freq, bsz//2)
            bwd_pos_emb = positional_embedding(bwd_pos_seq, inv_freq, bsz//2)
        else:
            fwd_pos_emb = positional_embedding(fwd_pos_seq, inv_freq)
            bwd_pos_emb = positional_embedding(bwd_pos_seq, inv_freq)

        pos_emb = tf.concat([fwd_pos_emb, bwd_pos_emb], axis=1)
    else:
        fwd_pos_seq = tf.range(beg, end, -1.0)
        if dtype is not None and dtype != tf.float32:
            fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
        if clamp_len > 0:
            fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -clamp_len, clamp_len)
        pos_emb = positional_embedding(fwd_pos_seq, inv_freq, bsz)

    return pos_emb


class MultiheadAttn(tf.Module):
    def __init__(self, d_model, n_head, d_head, dropout, dropatt, initializer,
                 is_training=True, residual=True, pre_ln=True, name=None, dtype=tf.float32):
        super().__init__(name=name)
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.dropout = dropout
        self.dropatt = dropatt
        self.initializer = initializer
        self.is_training = is_training
        self.residual = residual
        self.pre_ln = pre_ln
        self.dtype = dtype

    @tf.Module.with_name_scope
    def __call__(self, q, k, v, attn_mask):
        scale = 1 / (self.d_head ** 0.5)
        # Init layers
        if not hasattr(self, "q_proj"):
            self.q_proj = HeadProjection(
                self.d_model, self.n_head, self.d_head, self.initializer, name="q")
            self.k_proj = HeadProjection(
                self.d_model, self.n_head, self.d_head, self.initializer, name="k")
            self.v_proj = HeadProjection(
                self.d_model, self.n_head, self.d_head, self.initializer, name="v")
            self.post_attn = PostAttention(
                self.d_model, self.n_head, self.d_head, self.dropout, self.initializer,
                name="post_attn", residual=self.residual, pre_ln=self.pre_ln)

        v_inp = v

        if self.pre_ln:
            if not hasattr(self, "layer_norm_q"):
                self.layer_norm_q = tf.keras.layers.LayerNormalization(name='LayerNorm_q')
                self.layer_norm_k = tf.keras.layers.LayerNormalization(name='LayerNorm_k')
                self.layer_norm_v = tf.keras.layers.LayerNormalization(name='LayerNorm_v')

            q = self.layer_norm_q(q)
            k = self.layer_norm_k(k)
            v = self.layer_norm_v(v)

        q_head = self.q_proj(q)
        k_head = self.k_proj(k)
        v_head = self.v_proj(v)

        # attention vector
        attn_vec = abs_attn_core(q_head, k_head, v_head, attn_mask, self.dropatt,
                                 self.is_training, scale)
        # post processing
        output = self.post_attn(v_inp, attn_vec)

        return output


class RelMultiheadAttn(tf.Module):
    def __init__(self, d_model, n_head, d_head, dropout, dropatt, initializer,
                 is_training=True, residual=True, pre_ln=True, name=None, dtype=tf.float32):
        super().__init__(name=name)
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.dropout = dropout
        self.dropatt = dropatt
        self.initializer = initializer
        self.is_training = is_training
        self.residual = residual
        self.pre_ln = pre_ln
        self.dtype = dtype

    @tf.Module.with_name_scope
    def __call__(self, h, r, r_w_bias, r_r_bias, seg_mat, r_s_bias, seg_embed, attn_mask, mems):
        scale = 1 / (self.d_head ** 0.5)
        # Init layers
        self.q_proj = HeadProjection(
            self.d_model, self.n_head, self.d_head, self.initializer, name="q")
        self.k_proj = HeadProjection(
            self.d_model, self.n_head, self.d_head, self.initializer, name="k")
        self.v_proj = HeadProjection(
            self.d_model, self.n_head, self.d_head, self.initializer, name="v")
        self.r_proj = HeadProjection(
            self.d_model, self.n_head, self.d_head, self.initializer, name="r")
        self.post_attn = PostAttention(
            self.d_model, self.n_head, self.d_head, self.dropout, self.initializer,
            name="post_attn", residual=self.residual, pre_ln=self.pre_ln)

        hidden = h
        if self.pre_ln:
            self.layer_norm = tf.keras.layers.LayerNormalization(name='LayerNorm')
            hidden = self.layer_norm(hidden)

        if mems is not None and mems.shape.ndims > 1:
            cat = tf.concat([mems, hidden], 0)
        else:
            cat = hidden

        q_head_h = self.q_proj(hidden)
        k_head_h = self.k_proj(cat)
        v_head_h = self.v_proj(cat)
        k_head_r = self.r_proj(r)

        # core attention ops
        attn_vec = rel_attn_core(
                q_head_h, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat, r_w_bias,
                r_r_bias, r_s_bias, attn_mask, self.dropatt, self.is_training, scale)
        output = self.post_attn(h, attn_vec)

        return output


class LMLoss(tf.Module):
    def __init__(self, n_token, d_model, initializer, name=None, dtype=tf.float32):
        super().__init__(name=name)
        self.n_token = n_token
        self.d_model = d_model
        self.initializer = initializer
        self.dtype = dtype

    @tf.Module.with_name_scope
    def __call__(self, hidden, target, lookup_table=None):
        if lookup_table is None:
            if not hasattr(self, "weight"):
                self.weight = tf.Variable(
                    self.initializer([self.n_token, self.d_model], dtype=self.dtype),
                    name="weight"
                )
        else:
            self.weight = lookup_table
        if not hasattr(self, "bias"):
            self.bias = tf.Variable(
                self.initializer([self.n_token], dtype=self.dtype),
                name="bias"
            )

        self.logits = tf.einsum('ibd,nd->ibn', hidden, self.weight) + self.bias
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target,
            logits=self.logits)

        return self.loss, self.logits


class SummarizeSequence(tf.Module):
    def __init__(self, summary_type, d_model, n_head, d_head, dropout, dropatt, initializer,
                 is_training=True, residual=True, pre_ln=True, name=None, dtype=tf.float32,
                 use_proj=True):
        super().__init__(name=name)
        self.summary_type = summary_type
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.dropout = dropout
        self.dropatt = dropatt
        self.initializer = initializer
        self.is_training = is_training
        self.residual = residual
        self.pre_ln = pre_ln
        self.dtype = dtype
        self.use_proj = use_proj

    @tf.Module.with_name_scope
    def __call__(self, hidden, input_mask=None):
        if self.summary_type == "last":
            summary = hidden[-1]
        elif self.summary_type == "first":
            summary = hidden[0]
        elif self.summary_type == "mean":
            summary = tf.reduce_mean(hidden, axis=0)
        elif self.summary_type == "attn":
            # Init layers
            if not hasattr(self, "attn_layer"):
                self.attn_layer = MultiheadAttn(
                    self.d_model, self.n_head, self.d_head, self.dropout, self.dropatt,
                    self.initializer)
                self.summary_bias = tf.Variable(
                    self.initializer([self.d_model], dtype=self.dtype),
                    name="summary_bias")

            bsz = tf.shape(hidden)[1]
            bias = tf.tile(self.summary_bias[None, None], [1, bsz, 1])
            if input_mask is not None:
                input_mask = input_mask[None, :, :, None]

            summary = self.attn_layer(bias, hidden, hidden, input_mask)
            summary = summary[0]
        else:
            raise ValueError('Unsupported summary type {}'.format(self.summary_type))

        # use another projection as in BERT
        if self.use_proj:
            if not hasattr(self, "dense"):
                self.dense = tf.keras.layers.Dense(
                    self.d_model, activation=tf.nn.tanh, kernel_initializer=self.initializer,
                    name="proj")
            summary = self.dense(summary)

        if not hasattr(self, "dropout_layer"):
            self.dropout_layer = tf.keras.layers.Dropout(self.dropout, name="dropout_layer")
        summary = self.dropout_layer(summary, training=self.is_training)

        return summary


class ClassLoss(tf.Module):
    """
        Different classification tasks should use different scope names to ensure
        different dense layers (parameters) are used to produce the logits.

        An exception will be in transfer learning, where one hopes to transfer
        the classification weights.
    """
    def __init__(self, n_class, initializer, name=None):
        super().__init__(name=name)
        self.n_class = n_class
        self.initializer = initializer

    @tf.Module.with_name_scope
    def __call__(self, hidden, labels):
        # Init layers
        self.logit_layer = tf.keras.layers.Dense(
            self.n_class, kernel_initializer=self.initializer, name='logit_layer')

        logits = self.logit_layer(hidden)
        one_hot_target = tf.one_hot(labels, self.n_class, dtype=hidden.dtype)
        loss = -tf.reduce_sum(tf.nn.log_softmax(logits) * one_hot_target, -1)

        return loss, logits


def regression_loss(hidden, labels, initializer, scope, reuse=None,
                    return_logits=False):
    with tf.variable_scope(scope, reuse=reuse):
        logits = tf.layers.dense(
            hidden,
            1,
            kernel_initializer=initializer,
            name='logit')

        logits = tf.squeeze(logits, axis=-1)
        loss = tf.square(logits - labels)

        if return_logits:
            return loss, logits

        return loss
