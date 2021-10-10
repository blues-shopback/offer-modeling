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
