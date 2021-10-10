import tensorflow as tf

from models.transformer import modeling


class AttnLayer(tf.Module):
    def __init__(self, config, dtype, initializer, name="attn_layer", is_training=True):
        super().__init__(name=name)
        self.initializer = initializer
        self.config = config
        self.dtype = dtype
        self.is_training = is_training

    @tf.Module.with_name_scope
    def __call__(self, inp, inp_mask=None):
        config = self.config
        inputs = tf.transpose(inp, [1, 0, 2])
        # qlen = tf.shape(inputs)[0]

        if inp_mask is not None:
            input_mask = tf.transpose(inp_mask)
            input_mask = input_mask[None, :, :, None]

            attn_mask = tf.cast(input_mask > 0, dtype=self.dtype)
        else:
            attn_mask = None

        if not hasattr(self, "pre_proj_layer"):
            self.pre_proj_layer = tf.keras.layers.Dense(
                config.d_model, activation=None, kernel_initializer=self.initializer,
                name="pre-proj")
            self.pre_drop = tf.keras.layers.Dropout(config.dropout, name="pre-dropout")

        if not hasattr(self, "multi_attns"):
            self.multi_attns = []
            self.post_ffns = []
            self.layer_norms = []
            for i in range(config.n_layer):
                multi_attn = modeling.MultiheadAttn(
                    config.d_model, config.n_head, config.d_head, config.dropout, config.dropatt,
                    self.initializer, dtype=self.dtype, name="transformer_{}".format(i))
                position_ffn = modeling.PositionwiseFFN(
                    config.d_model, config.d_inner, config.dropout, self.initializer,
                    is_training=self.is_training, name="post_ffn_{}".format(i))
                layer_norm1 = tf.keras.layers.LayerNormalization(
                    name='LayerNorm1_{}'.format(i), axis=-1)
                layer_norm2 = tf.keras.layers.LayerNormalization(
                    name='LayerNorm2_{}'.format(i), axis=-1)

                self.multi_attns.append(multi_attn)
                self.post_ffns.append(position_ffn)
                self.layer_norms.append((layer_norm1, layer_norm2))

        output_h = inputs
        output_h = self.pre_proj_layer(output_h)
        output_h = self.pre_drop(output_h, training=self.is_training)

        for attn, ffn, lnorm_tuple in zip(self.multi_attns, self.post_ffns, self.layer_norms):
            ln1, ln2 = lnorm_tuple
            if config.pre_ln:
                output_ori = output_h
                output_h = ln1(output_h)
                output_h = attn(output_h, output_h, output_h, attn_mask)
                output_h = output_h + output_ori

                output_ori = output_h
                output_h = ln2(output_h)
                output_h = ffn(output_h)
                output_h = output_h + output_ori
            else:
                output_ori = output_h
                output_h = attn(output_h, output_h, output_h, attn_mask)
                output_h = ln1(output_h + output_ori)

                output_ori = output_h
                output_h = ffn(output_h)
                output_h = ln2(output_h + output_ori)

        if config.pre_ln:
            if not hasattr(self, "layer_norm_out"):
                self.layer_norm_out = tf.keras.layers.LayerNormalization(name='LayerNorm_out')
            output_h = self.layer_norm_out(output_h)

        return output_h


class BaseModel(tf.Module):
    def __init__(self, config, initializer=None, name="base_model", dtype=tf.float32,
                 is_training=True):
        super().__init__(name=name)
        self.config = config
        if initializer is None:
            self.initializer = tf.keras.initializers.TruncatedNormal(
                mean=0.0, stddev=0.05, seed=None)
        else:
            self.initializer = initializer
        self.dtype = dtype
        self.is_training = is_training

    @tf.Module.with_name_scope
    def __call__(self, inp, ids, inp_mask=None):
        """
            Args:
                inp: shape 4D [bsz, qlen, value_num, 3(value, log, sign)]
                ids: shape 1D [bsz]
        """
        config = self.config
        bsz = tf.shape(inp)[0]
        qlen = tf.shape(inp)[1]
        value_num = tf.shape(inp)[2]
        value_d = tf.shape(inp)[3]
        inp_reshape = tf.reshape(inp, shape=[-1, value_num, value_d])
        if not hasattr(self, "attn_layer"):
            self.attn_layer = AttnLayer(config, self.dtype, is_training=self.is_training)
            self.summarize_attn_layer = modeling.SummarizeSequence(
                "attn", config.d_model, config.n_head, config.d_head, config.dropout,
                config.dropatt, self.initializer, name="summarize_attn_layer")

        output = self.attn_layer(inp_reshape, inp_mask)
        output = self.summarize_attn_layer(output, input_mask=inp_mask)

        # reshape back to [bsz, qlen, d_model]
        inputs = tf.reshape(output, [bsz, qlen, config.d_model])
        inputs = tf.transpose(inputs, [1, 0, 2])

        if not hasattr(self, "cate_emb_layer"):
            self.cate_emb_layer = modeling.Embedding(
                config.n_ticker, config.d_embed, self.initializer, name="cate_emb_layer")

        ids = ids[:, None]
        cate_embedding = self.cate_emb_layer(ids)
        cate_embedding = tf.tile(cate_embedding, [1, qlen, 1])
        cate_embedding = tf.transpose(cate_embedding, [1, 0, 2])
        # bsz = tf.shape(inputs)[1]
        causal_mask = modeling._create_mask(qlen, 0, self.dtype, False)
        causal_mask = causal_mask[:, :, None, None]

        attn_mask = causal_mask

        if inp_mask is not None:
            input_mask = tf.transpose(inp_mask)
            input_mask = input_mask[None, :, :, None]
            attn_mask += input_mask

        attn_mask = tf.cast(attn_mask > 0, dtype=self.dtype)

        # self.embedding_layer = modeling.Embedding(
        #     config.n_token, config.d_embed, self.initializer, name="embedding",
        #     dtype=self.dtype)

        # emb = self.embedding_layer(inputs)

        # position embedding
        # pos_emb = modeling.positional_encoding(
        #     qlen, config.d_model, -1, bsz=bsz, dtype=self.dtype)

        output_h = tf.concat([inputs, cate_embedding], axis=-1)

        if not hasattr(self, "pre_proj_layer"):
            self.pre_proj_layer = tf.keras.layers.Dense(
                config.d_model, activation=None, kernel_initializer=self.initializer,
                name="pre-proj")
            self.pre_drop = tf.keras.layers.Dropout(config.dropout, name="pre-dropout")
        output_h = self.pre_proj_layer(output_h)
        output_h = self.pre_drop(output_h, training=self.is_training)

        if not hasattr(self, "multi_attns"):
            self.multi_attns = []
            self.post_ffns = []
            for i in range(config.n_layer):
                multi_attn = modeling.MultiheadAttn(
                    config.d_model, config.n_head, config.d_head, config.dropout, config.dropatt,
                    self.initializer, dtype=self.dtype, name="transformer_{}".format(i))
                position_ffn = modeling.PositionwiseFFN(
                    config.d_model, config.d_inner, config.dropout, self.initializer,
                    is_training=self.is_training, name="post_ffn_{}".format(i))

                self.multi_attns.append(multi_attn)
                self.post_ffns.append(position_ffn)

        for attn, ffn in zip(self.multi_attns, self.post_ffns):
            output_h = attn(output_h, output_h, output_h, attn_mask)
            output_h = ffn(output_h)

        if config.pre_ln:
            if not hasattr(self, "layer_norm"):
                self.layer_norm = tf.keras.layers.LayerNormalization(name='LayerNorm_out')
            output_h = self.layer_norm(output_h)

        if not hasattr(self, "summarize_layer"):
            self.summarize_layer = modeling.SummarizeSequence(
                "attn", config.d_model, config.n_head, config.d_head, config.dropout,
                config.dropatt, self.initializer)

        summary = self.summarize_layer(output_h, input_mask=inp_mask)

        return summary
