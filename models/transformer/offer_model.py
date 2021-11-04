import tensorflow as tf

from models.transformer import modules, modeling, base_bert


class OfferModel(tf.Module):
    def __init__(self, config, initializer=None, name="offer_model", dtype=tf.float32,
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

    # @tf.Module.with_name_scope
    # def get_contrastive_loss(self, pooled, ):

    @tf.Module.with_name_scope
    def get_mlm_loss(self, output, target, mlm_pos_padded):
        masked_target = target * mlm_pos_padded
        loss, logits = self.encoder.masked_lm_loss(output, masked_target)

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
                use_proj=True
            )

        output = self.encoder(inp, pos_cate, inp_mask)
        pooled = self.pooler(output, inp_mask)

        return pooled, output
