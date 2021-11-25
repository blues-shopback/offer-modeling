import tensorflow as tf

from models.transformer import modeling


class SummarizeSequence(tf.Module):
    def __init__(self, summary_type, d_model, n_head, d_head, dropout, dropatt, initializer,
                 residual=True, name=None, dtype=tf.float32, use_proj=True, is_training=True):
        super().__init__(name=name)
        self.summary_type = summary_type
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.dropout = dropout
        self.dropatt = dropatt
        self.initializer = initializer
        self.residual = residual
        self.dtype = dtype
        self.use_proj = use_proj
        self.is_training = is_training

    @tf.Module.with_name_scope
    def __call__(self, hidden, input_mask=None):
        """
        Args:
            hidden: shape 3D [bsz, qlen, d_model]
        """
        hidden = tf.transpose(hidden, [1, 0, 2])
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

            summary = self.attn_layer(bias, hidden, hidden, input_mask, self.is_training)
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


class MultiheadAttn(tf.Module):
    def __init__(self, d_model, n_head, d_head, dropout, dropatt, initializer,
                 name=None, dtype=tf.float32):
        super().__init__(name=name)
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.dropout = dropout
        self.dropatt = dropatt
        self.initializer = initializer
        self.dtype = dtype

    @tf.Module.with_name_scope
    def __call__(self, q, k, v, attn_mask, is_training):
        """q, k, v dims: [seq, batch, emb]"""
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
                name="post_attn")

        v_inp = v

        q_head = self.q_proj(q)
        k_head = self.k_proj(k)
        v_head = self.v_proj(v)

        # attention vector
        if not hasattr(self, "attn_dropout"):
            self.attn_dropout = tf.keras.layers.Dropout(self.dropatt, name="attn_dropout")
        attn_vec = modeling.abs_attn_core(q_head, k_head, v_head, attn_mask, self.attn_dropout,
                                          is_training, scale)
        # post processing
        output = self.post_attn(v_inp, attn_vec)

        return output


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
                self.initializer([self.n_token, self.d_embed], dtype=self.dtype),
                name="embedding")

        return tf.nn.embedding_lookup(self.lookup_table, x)


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
                 is_training=True, name=None, dtype=tf.float32):
        super().__init__(name=name)
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.dropout = dropout
        self.initializer = initializer
        self.is_training = is_training
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

        return attn_out


class PositionWiseFFN(tf.Module):
    def __init__(self, d_model, d_inner, dropout, initializer,
                 activation_type='relu', name='ff', is_training=True):
        super().__init__(name=name)
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.initializer = initializer
        self.activation_type = activation_type
        self.is_training = is_training
        if self.activation_type == 'relu':
            self.act = tf.nn.relu
        elif self.activation_type == 'gelu':
            self.act = modeling.gelu
        else:
            raise ValueError('Unsupported activation type {}'.format(self.activation_type))

    @tf.Module.with_name_scope
    def __call__(self, inp):
        """Position-wise Feed-forward Network."""
        output = inp

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

        return output
