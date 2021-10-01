import tensorflow as tf
import tensorflow_probability as tfp

from models import config_utils
from models.transformer import modeling
from models.normalize_flow import realnvp


class FlowFinConfig(config_utils.BaseConfig):
    @staticmethod
    def get_keys():
        return ["n_layer", "d_model", "d_embed", "n_head", "d_head", "d_inner",
                "dropout", "dropatt", "pre_ln",
                "n_ticker",
                "num_realnvp", "realnvp_hidden_layers", "sample_size"]


def build_realnvp_flow(dist, num_realnvp, output_dim, hidden_layers, name="realnvp"):
    bijector_chain = []
    r = list(range(output_dim))
    r1 = r[:(output_dim//2)]
    r2 = r[(output_dim//2):]
    perm = r2 + r1

    for i in range(num_realnvp):
        bijector_chain.append(
            realnvp.RealNVP(
                output_dim, hidden_layers,
                forward_min_event_ndims=1, name=name+"_{}".format(i)))
        bijector_chain.append(tfp.bijectors.Permute(perm))
    chain = tfp.bijectors.Chain(list(reversed(bijector_chain)))
    flow = tfp.distributions.TransformedDistribution(
        distribution=dist,
        bijector=chain)

    return flow


class FinModel(tf.Module):
    def __init__(self, config, dtype, name="fin_model", is_training=True):
        super().__init__(name=name)
        self.config = config
        self.initializer = tf.keras.initializers.TruncatedNormal(
            mean=0.0, stddev=0.05, seed=None)
        self.dtype = dtype
        self.is_training = is_training

    @tf.Module.with_name_scope
    def binary_proj(self, inp):
        if not hasattr(self, "binary_layer"):
            self.binary_layer = tf.keras.layers.Dense(
                1, kernel_initializer=self.initializer, name='binary_layer')
        logits = self.binary_layer(inp)

        return logits

    def get_binary_loss(self, logits, target):
        loss_batch = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=target, logits=logits, name="binary_loss")

        return loss_batch

    def get_flow_loss(self):
        return self.flow_loss

    @tf.Module.with_name_scope
    def __call__(self, inp, ids, inp_mask=None):
        """
            Args:
                inp: shape 4D [bsz, qlen, value_num, 3(value, log, sign)]
                ids: shape 1D [bsz]
        """
        config = self.config
        qlen = tf.shape(inp)[1]

        if not hasattr(self, "cate_emb_layer"):
            self.cate_emb_layer = modeling.Embedding(
                config.n_ticker, config.d_embed, self.initializer, name="cate_emb_layer")

        ids = ids[:, None]
        cate_embedding = self.cate_emb_layer(ids)
        cate_embedding = tf.tile(cate_embedding, [1, qlen, 1])
        cate_embedding = tf.transpose(cate_embedding, [1, 0, 2])

        if not hasattr(self, "realnvp_layers"):
            self.realnvp_layers = FinRealNVPModel(
                self.config, self.dtype, name="fin_realnvp", is_training=self.is_training)
        # shape: [sample, bsz, seq, d_model]
        flow1_out, loglikeihood = self.realnvp_layers(inp, sample_size=config.sample_size)
        self.flow_loss = -tf.reduce_mean(loglikeihood)

        flow1_out = tf.reduce_mean(flow1_out, axis=0)
        flow1_out = tf.transpose(flow1_out, [1, 0, 2])  # [s, b, d]
        output_h = tf.concat([flow1_out, cate_embedding], axis=-1)

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

        causal_mask = modeling._create_mask(qlen, 0, self.dtype, False)
        causal_mask = causal_mask[:, :, None, None]
        attn_mask = causal_mask
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


class FinRealNVPModel(tf.Module):
    def __init__(self, config, dtype, name="fin_realnvp", is_training=True):
        super().__init__(name=name)
        self.initializer = tf.keras.initializers.TruncatedNormal(
            mean=0.0, stddev=0.05, seed=None)
        self.config = config
        self.dtype = dtype
        self.is_training = is_training

    def _norm_diag_sample(self, sample_size, loc, scale):
        loc_shape = tf.shape(loc)
        shape = tf.concat([[sample_size], loc_shape], axis=-1)
        norm_sample = tf.random.normal(shape)

        sample = loc[None] + scale[None] * norm_sample

        return sample

    def _norm_log_prob(self, x, loc, scale):
        sqrt_2pi = tf.constant(2.5066282746310002)
        mu = loc[None]
        sigma = scale[None]
        logp = -0.5 * tf.square((x - mu) / sigma) - tf.math.log(sigma * sqrt_2pi)

        return logp

    @tf.Module.with_name_scope
    def __call__(self, inp, inp_mask=None, sample_size=1):
        config = self.config
        bsz = tf.shape(inp)[0]
        qlen = tf.shape(inp)[1]
        value_num = tf.shape(inp)[2]
        value_d = tf.shape(inp)[3]
        if not hasattr(self, "loc_scale_layer"):
            self.loc_scale_layer = realnvp.LocScaleLayer(
                config, self.dtype, is_training=self.is_training
            )
        # reshape to [bsz*qlen, value_num, 3]
        inp_reshape = tf.reshape(inp, shape=[-1, value_num, value_d])

        loc, scale = self.loc_scale_layer(inp_reshape, inp_mask=inp_mask)

        # reshape back to [bsz, qlen, d_model]
        loc = tf.reshape(loc, [bsz, qlen, config.d_model])
        scale = tf.reshape(scale, [bsz, qlen, config.d_model])

        loc_next = loc[:, 1:, :]
        scale_next = scale[:, 1:, :]
        loc_last = loc[:, :-1, :]
        scale_last = scale[:, :-1, :]

        if not hasattr(self, "flow1"):
            self.flow1 = realnvp.RealNVP2(
                config.num_realnvp,
                config.d_model,
                config.realnvp_hidden_layers,
                name="realnvp1")
            self.flow2 = realnvp.RealNVP2(
                config.num_realnvp,
                config.d_model,
                config.realnvp_hidden_layers,
                name="realnvp2")

        flow1_out, log_s1 = self.flow1(loc, scale, direction="forward", sample_size=sample_size)
        flow2_out, log_s2 = self.flow2(loc, scale, direction="forward", sample_size=sample_size)

        # Loss
        if self.is_training:
            flow1_2_base, flow1_2_logs = self.flow2(
                None, None, direction="invert", inp=flow1_out[:, :, :-1, :])
            logp1_2 = self.flow2._norm_log_prob(flow1_2_base, loc_next, scale_next)
            flow1_2_loglikeihood = \
                tf.reduce_sum(logp1_2, axis=-1) - tf.reduce_sum(flow1_2_logs, axis=-1)

            flow2_1_base, flow2_1_logs = self.flow1(
                None, None, direction="invert", inp=flow2_out[:, :, 1:, :])
            logp2_1 = self.flow1._norm_log_prob(flow2_1_base, loc_last, scale_last)
            flow2_1_loglikeihood = \
                tf.reduce_sum(logp2_1, axis=-1) - tf.reduce_sum(flow2_1_logs, axis=-1)

            loglikeihood = flow1_2_loglikeihood + flow2_1_loglikeihood

            return flow1_out, loglikeihood

        else:
            return flow1_out
