import tensorflow as tf

from models.transformer import modules, base_bert


class OfferModel(tf.Module):
    def __init__(self, config, initializer=None, name="offer_model", dtype=tf.float32,
                 cate_size=None, is_training=True):
        super().__init__(name=name)
        self.config = config
        self.cate_size = cate_size
        if initializer is None:
            self.initializer = tf.keras.initializers.TruncatedNormal(
                mean=0.0, stddev=0.02, seed=None)
        else:
            self.initializer = initializer
        self.dtype = dtype
        self.is_training = is_training

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

        # shape [any]
        pos_cos = tf.gather_nd(self_cossim, pos_pair_idx)
        # shape [any, bsz]
        others_cos = tf.gather_nd(self_cossim, pos_pair_idx[:, 0:1])

        # cross-entropy
        softmax_denominator = tf.reduce_logsumexp(others_cos, axis=1)

        contrastive_loss = -1 * (pos_cos - softmax_denominator)

        return contrastive_loss

    @tf.Module.with_name_scope
    def get_mlm_loss(self, output, target):
        loss, logits = self.encoder.masked_lm_loss(output, target)

        return loss

    @tf.Module.with_name_scope
    def get_cate_loss(self, target):
        logits = tf.einsum('bd,nd->bn', self.pooled_cate, self.cate_w) + self.cate_b
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                              logits=logits)

        return loss

    @tf.Module.with_name_scope
    def __call__(self, inp, pos_cate=None, inp_mask=None):
        config = self.config

        if not hasattr(self, "encoder"):
            self.encoder = base_bert.BaseModel(
                config, self.initializer, name="bert_encoder", dtype=self.dtype,
                is_training=self.is_training)

            self.pooler = modules.SummarizeSequence(
                summary_type="attn",
                d_model=config.d_model,
                n_head=config.n_head,
                d_head=config.d_head,
                dropout=config.dropout,
                dropatt=config.dropatt,
                initializer=self.initializer,
                is_training=self.is_training,
                name="attn_pooler",
                dtype=self.dtype,
                use_proj=self.is_training
            )
            if self.cate_size is not None:
                self.cate_pooler = modules.SummarizeSequence(
                    summary_type="attn",
                    d_model=config.d_model,
                    n_head=config.n_head,
                    d_head=config.d_head,
                    dropout=config.dropout,
                    dropatt=config.dropatt,
                    initializer=self.initializer,
                    is_training=self.is_training,
                    name="attn_pooler_cate",
                    dtype=self.dtype,
                    use_proj=True
                )
                self.cate_w = tf.Variable(
                    self.initializer([self.cate_size, config.d_model], dtype=self.dtype),
                    name="cate_weight"
                )
                self.cate_b = tf.Variable(
                    self.initializer([self.cate_size], dtype=self.dtype),
                    name="cate_bias"
                )

        output = self.encoder(inp, pos_cate, inp_mask)
        pooled = self.pooler(output, inp_mask)

        if self.cate_size is not None:
            self.pooled_cate = self.cate_pooler(output, inp_mask)

        return pooled, output
