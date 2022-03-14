import tensorflow as tf

from models.transformer import modules, base_bert


class OfferModel(tf.Module):
    def __init__(self, config, initializer=None, name="offer_model", dtype=tf.float32,
                 is_training=True, add_pooler=True):
        super().__init__(name=name)
        self.config = config
        self.add_pooler = add_pooler
        if initializer is None:
            self.initializer = tf.keras.initializers.TruncatedNormal(
                mean=0.0, stddev=0.02, seed=None)
        else:
            self.initializer = initializer
        self.dtype = dtype
        self.is_training = is_training
        self.built_cate = False

    def get_trainable_vars(self, merchant):
        train_vars = []
        train_vars += self.encoder.trainable_variables
        if self.add_pooler:
            train_vars += self.pooler.trainable_variables
        if self.built_cate:
            pooler, cate_w, cate_b = self.merchant_classify_layey_map[merchant]
            train_vars += pooler.trainable_variables
            train_vars.append(cate_w)
            train_vars.append(cate_b)

        return train_vars

    @tf.function
    def traced_call(self, inp, pos_cate, inp_mask):
        return self.__call__(inp, pos_cate, inp_mask)

    @tf.Module.with_name_scope
    def get_contrastive_loss(self, pooled, pos_pair_idx, temp=1.0):
        """
        Args:
        pooled: shape 2D [bsz, d_model]
            pooled logits.
        pos_pair_idx: shape 2D [any, 2]
            positive pair index in batch.
            - ex: [0, 1] indicate pooled[0, :] and pooled[1, :] are positive pair.
        temp: scalar
            temperature for softmax.
        """
        normalize_logits = tf.math.l2_normalize(pooled, axis=-1)
        # shape [bsz, bsz]
        self_cossim = tf.einsum("id,jd->ij", normalize_logits, normalize_logits)
        # temperature for softmax
        self_cossim = self_cossim / temp

        # positive pair cossim: shape [any]
        pos_cos = tf.gather_nd(self_cossim, pos_pair_idx)

        # extract softmax_denominator
        # Remove diag value
        bsz = tf.shape(pooled)[0]
        diag_ones = tf.ones([bsz, bsz], dtype=self.dtype)
        # Set diag to 0
        bool_mask = tf.linalg.set_diag(
            diag_ones,
            tf.zeros([bsz], dtype=self.dtype)
        )
        # Set positive pair to 0
        bool_mask = tf.tensor_scatter_nd_add(
            bool_mask,
            pos_pair_idx,
            -tf.ones(tf.shape(pos_pair_idx)[0], dtype=tf.float32)
        )
        # gather mask and value from target batch index
        gather_self_cossim = tf.gather_nd(self_cossim, pos_pair_idx[:, 0:1])
        gather_bool_mask = tf.gather_nd(bool_mask, pos_pair_idx[:, 0:1])
        # Exclude position in diag and positive pair
        extract_cossim = tf.boolean_mask(gather_self_cossim, gather_bool_mask)
        # shape [any, bsz - 2]
        extract_cossim = tf.reshape(
            extract_cossim,
            [tf.shape(gather_self_cossim)[0], bsz - 2]
        )

        # cross-entropy
        softmax_denominator = tf.reduce_logsumexp(extract_cossim, axis=1)

        contrastive_loss = -1 * (pos_cos - softmax_denominator)

        return contrastive_loss

    @tf.Module.with_name_scope
    def get_mlm_loss(self, output, target):
        loss, logits = self.encoder.masked_lm_loss(output, target)

        return loss

    @tf.Module.with_name_scope
    def get_cate_loss(self, output, inp_mask, target, merchant):
        """Ignore target < 0"""
        assert self.built_cate, "Need to call 'build_classify_layer' before 'get_cate_loss'."

        mock_target = tf.zeros_like(target)

        mask_pos = tf.where(tf.math.greater_equal(target, 0), 1., 0.)
        masked_target = tf.where(tf.math.greater_equal(target, 0), target, 0)

        losses = 0.

        for merc in self.merchant_classify_layey_map:
            pool_layer, cate_w, cate_b = self.merchant_classify_layey_map[merc]

            m_target = tf.cond(
                tf.equal(merchant, merc), lambda: masked_target, lambda: mock_target)

            pooled = pool_layer(output, inp_mask)
            logits = tf.einsum('bd,nd->bn', pooled, cate_w) + cate_b

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=m_target,
                logits=logits)

            mask_loss = loss * mask_pos

            loss_sum = tf.reduce_sum(mask_loss)
            loss_dom = tf.math.maximum(tf.reduce_sum(mask_pos), 1.)
            reduce_loss = loss_sum / loss_dom

            weighted_loss = reduce_loss * tf.where(
                tf.equal(merchant, merc), 1., 0.)

            losses += weighted_loss

        return losses

    def create_classify_layer(self, name, cate_size):
        config = self.config
        pool_layer = modules.SummarizeSequence(
            summary_type="attn",
            d_model=config.d_model,
            n_head=config.n_head,
            d_head=config.d_head,
            dropout=config.dropout,
            dropatt=config.dropatt,
            initializer=self.initializer,
            is_training=self.is_training,
            name="attn_pooler_{}".format(name),
            dtype=self.dtype,
            use_proj=True
        )
        cate_w = tf.Variable(
            self.initializer([cate_size, config.d_model], dtype=self.dtype),
            name="cate_weight_{}".format(name)
        )
        cate_b = tf.Variable(
            self.initializer([cate_size], dtype=self.dtype),
            name="cate_bias_{}".format(name)
        )
        return pool_layer, cate_w, cate_b

    @tf.Module.with_name_scope
    def build_classify_layer(self, merchant_and_cate_size_list):
        """
        args:
            merchant_and_cate_size_list: tuple of merchant name and category size.
            ex: [(amazon, 300),]
        """
        if not self.built_cate:
            self.merchant_classify_layey_map = {}
            for merchant, cate_size in merchant_and_cate_size_list:
                pool_layer, cate_w, cate_b = self.create_classify_layer(
                    merchant, cate_size)
                self.merchant_classify_layey_map[merchant] = (pool_layer, cate_w, cate_b)
                setattr(self, merchant, pool_layer)

            self.built_cate = True

    @tf.Module.with_name_scope
    def __call__(self, inp, pos_cate=None, inp_mask=None):
        config = self.config

        if not hasattr(self, "encoder"):
            self.encoder = base_bert.BaseModel(
                config, self.initializer, name="bert_encoder", dtype=self.dtype,
                is_training=self.is_training)
            if self.add_pooler:
                self.pooler = modules.SummarizeSequence(
                    summary_type=config.summary_type,
                    d_model=config.d_model,
                    n_head=config.n_head,
                    d_head=config.d_head,
                    dropout=config.dropout,
                    dropatt=config.dropatt,
                    initializer=self.initializer,
                    is_training=self.is_training,
                    name="attn_pooler",
                    dtype=self.dtype,
                    use_proj=True
                )

        output = self.encoder(inp, pos_cate, inp_mask)

        if self.add_pooler:
            pooled = self.pooler(output, inp_mask)
        else:
            pooled = None

        return pooled, output
