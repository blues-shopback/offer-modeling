import tensorflow as tf

from models.transformer import modules, modeling


class LMLoss(tf.Module):
    def __init__(self, n_token, initializer=None, lookup_table=None, name="lm_loss"):
        super().__init__(name=name)
        self.n_token = n_token
        self.lookup_table = lookup_table
        if initializer is None:
            self.initializer = tf.keras.initializers.TruncatedNormal(
                mean=0.0, stddev=0.02, seed=None)
        else:
            self.initializer = initializer

    @tf.Module.with_name_scope
    def __call__(self, hidden, target):
        if not hasattr(self, "softmax_b"):
            self.softmax_b = tf.Variable(
                self.initializer([self.n_token], dtype=hidden.dtype),
                name="softmax_bias"
            )
            if self.lookup_table is None:
                self.softmax_w = tf.Variable(
                    self.initializer([self.n_token, hidden.shape[-1]], dtype=hidden.dtype),
                    name="softmax_weight"
                )
            else:
                self.softmax_w = self.lookup_table

        logits = tf.einsum('bd,nd->bn', hidden, self.softmax_w) + self.softmax_b
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                              logits=logits)
        return loss, logits


class AttnLayer(tf.Module):
    def __init__(self, config, dtype, initializer, name="attn_layer", is_training=True):
        super().__init__(name=name)
        self.initializer = initializer
        self.config = config
        self.dtype = dtype
        self.is_training = is_training

    @tf.Module.with_name_scope
    def __call__(self, inp, inp_mask=None):
        """inp shape: [qlen, bsz, d_model]"""
        config = self.config

        if inp_mask is not None:
            input_mask = tf.transpose(inp_mask)
            input_mask = input_mask[None, :, :, None]

            attn_mask = tf.cast(input_mask > 0, dtype=self.dtype)
        else:
            attn_mask = None

        if not hasattr(self, "pre_proj_layer"):
            self.pre_proj_layer = tf.keras.layers.Dense(
                config.d_model, activation=None, kernel_initializer=self.initializer,
                name="pre_proj")
            self.pre_drop = tf.keras.layers.Dropout(config.dropout, name="pre_dropout")

        if not hasattr(self, "multi_attns"):
            self.multi_attns = []
            self.post_ffns = []
            self.layer_norms = []
            for i in range(config.n_layer):
                multi_attn = modules.MultiheadAttn(
                    config.d_model, config.n_head, config.d_head, config.dropout, config.dropatt,
                    self.initializer, dtype=self.dtype, name="transformer_{}".format(i))
                position_ffn = modules.PositionWiseFFN(
                    config.d_model, config.d_inner, config.dropout, self.initializer,
                    is_training=self.is_training, name="post_ffn_{}".format(i))
                layer_norm1 = tf.keras.layers.LayerNormalization(
                    name='LayerNorm1_{}'.format(i), axis=-1)
                layer_norm2 = tf.keras.layers.LayerNormalization(
                    name='LayerNorm2_{}'.format(i), axis=-1)

                self.multi_attns.append(multi_attn)
                self.post_ffns.append(position_ffn)
                self.layer_norms.append((layer_norm1, layer_norm2))

        output_h = inp
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
                mean=0.0, stddev=0.02, seed=None)
        else:
            self.initializer = initializer
        self.dtype = dtype
        self.is_training = is_training

    @tf.Module.with_name_scope
    def masked_lm_loss(self, output, target):
        """
        Args:
            output: shape 3D [bsz, qlen, d_model]
            target: shape 2D [bsz, qlen]
                only eval on posistion value >= 0
                ex: [-5643, 0, 43, -78], will eval 0, and 43.
        """
        if not hasattr(self, "lm_loss_layer"):
            lookup_table = self.embedding_layer.lookup_table
            self.lm_loss_layer = LMLoss(
                self.config.n_token, self.initializer, lookup_table, name="lm_loss")
        output_masked, target_masked = modeling.gather_for_masked_lm_loss(output, target)
        output_masked = tf.reshape(output_masked, [-1, self.config.d_model])

        loss, logits = self.lm_loss_layer(output_masked, target_masked)

        return loss, logits

    @tf.Module.with_name_scope
    def __call__(self, inp, pos_cate=None, inp_mask=None):
        """
        Args:
            inp: shape 2D [bsz, qlen]
            pos_cate: shape 2D [bsz, qlen]
            inp_mask: shape 2D [bsz, qlen]
                - 0 for using, 1 for masking.
        """
        config = self.config
        bsz = tf.shape(inp)[0]
        qlen = tf.shape(inp)[1]
        if not hasattr(self, "attn_layer"):
            if pos_cate is not None:
                self.inp_cate_embedding_layer = modules.Embedding(
                    config.inp_cate_num, config.d_embed, self.initializer, dtype=self.dtype,
                    name="embedding_inp_cate")
            self.embedding_layer = modules.Embedding(
                config.n_token, config.d_embed, self.initializer, dtype=self.dtype,
                name="embedding")
            self.word_emb_dropout = tf.keras.layers.Dropout(
                config.dropout, name="word_emb_dropout")
            self.attn_layer = AttnLayer(
                config, self.dtype, self.initializer,
                is_training=self.is_training)
            # self.summarize_attn_layer = modules.SummarizeSequence(
            #     "attn", config.d_model, config.n_head, config.d_head, config.dropout,
            #     config.dropatt, self.initializer, name="summarize-attn-layer")

        inp_trans = tf.transpose(inp, [1, 0])
        word_emb = self.embedding_layer(inp_trans)
        word_emb = self.word_emb_dropout(word_emb, training=self.is_training)
        pos_emb = modeling.positional_encoding(
            qlen, config.d_embed, clamp_len=-1, bsz=bsz, dtype=self.dtype)

        if pos_cate is None:
            output = word_emb + pos_emb
        else:
            pos_cate_trans = tf.transpose(pos_cate, [1, 0])
            inp_cate_emb = self.inp_cate_embedding_layer(pos_cate_trans)
            output = word_emb + pos_emb + inp_cate_emb

        output = self.attn_layer(output, inp_mask)
        output = tf.transpose(output, [1, 0, 2])

        return output  # [bsz, qlen, d_model]
