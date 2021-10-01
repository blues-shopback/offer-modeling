import tensorflow as tf

from models.transformer import modeling


class TestModel(tf.Module):
    def __init__(self, config, name="test_model", dtype=tf.float32, is_training=True):
        super().__init__(name=name)
        self.config = config
        self.initializer = tf.keras.initializers.TruncatedNormal(
            mean=0.0, stddev=0.05, seed=None)
        self.dtype = dtype
        self.is_training = is_training

    @tf.Module.with_name_scope
    def __call__(self, inp, inp_mask=None):
        config = self.config

        inputs = tf.transpose(inp, [1, 0, 2])
        qlen = tf.shape(inputs)[0]
        bsz = tf.shape(inputs)[1]
        causal_mask = modeling._create_mask(qlen, 0, self.dtype, False)
        causal_mask = causal_mask[:, :, None, None]
        causal_mask = tf.cast(causal_mask > 0, dtype=self.dtype)

        if inp_mask is not None:
            input_mask = tf.transpose(inp_mask)
            input_mask = input_mask[None, :, :, None]
            attn_mask = causal_mask + input_mask
        else:
            attn_mask = causal_mask

        # self.embedding_layer = modeling.Embedding(
        #     config.n_token, config.d_embed, self.initializer, name="embedding",
        #     dtype=self.dtype)

        # emb = self.embedding_layer(inputs)
        emb = inputs

        if config.d_embed != config.d_model:
            if not hasattr(self, "proj_layer"):
                self.proj_layer = tf.keras.layers.Dense(
                    config.d_model, activation=None, kernel_initializer=self.initializer,
                    name="proj")
            emb = self.proj_layer(emb)

        # position embedding
        pos_emb = modeling.positional_encoding(
            qlen, config.d_model, -1, bsz=bsz, dtype=self.dtype)

        output_h = emb + pos_emb

        if not hasattr(self, "mult_attns"):
            self.mult_attns = []
            self.post_ffns = []
            for i in range(config.n_layer):
                mult_attn = modeling.MultiheadAttn(
                    config.d_model, config.n_head, config.d_head, config.dropout, config.dropatt,
                    self.initializer, dtype=self.dtype, name="transformer_{}".format(i))
                position_ffn = modeling.PositionwiseFFN(
                    config.d_model, config.d_inner, config.dropout, self.initializer,
                    is_training=self.is_training, name="post_ffn_{}".format(i))

                self.mult_attns.append(mult_attn)
                self.post_ffns.append(position_ffn)

        for attn, ffn in zip(self.mult_attns, self.post_ffns):
            output_h = attn(output_h, output_h, output_h, attn_mask)
            output_h = ffn(output_h)

        if config.pre_ln:
            if not hasattr(self, "layer_norm"):
                self.layer_norm = tf.keras.layers.LayerNormalization(name='LayerNorm_out')
            output_h = self.layer_norm(output_h)

        return output_h
