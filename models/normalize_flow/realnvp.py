import tensorflow as tf
import tensorflow_probability as tfp

# from models import config_utils
from models.transformer import modeling


class LocScaleLayer(tf.Module):
    def __init__(self, config, dtype, name="loc_scale_layer", is_training=True):
        super().__init__(name=name)
        self.initializer = tf.keras.initializers.TruncatedNormal(
            mean=0.0, stddev=0.05, seed=None)
        self.config = config
        self.dtype = dtype
        self.is_training = is_training

    @tf.Module.with_name_scope
    def __call__(self, inp, inp_mask=None):
        config = self.config
        if not hasattr(self, "attn_layer"):
            self.attn_layer = AttnLayer(config, self.dtype, is_training=self.is_training)

        output = self.attn_layer(inp, inp_mask)

        if not hasattr(self, "summarize_loc_layer"):
            self.summarize_loc_layer = modeling.SummarizeSequence(
                "attn", config.d_model, config.n_head, config.d_head, config.dropout,
                config.dropatt, self.initializer, name="summarize_loc_layer")
            self.summarize_scale_layer = modeling.SummarizeSequence(
                "attn", config.d_model, config.n_head, config.d_head, config.dropout,
                config.dropatt, self.initializer, name="summarize_scale_layer")
        loc = self.summarize_loc_layer(output, input_mask=inp_mask)
        loc = tf.math.tanh(loc)
        scale = self.summarize_scale_layer(output, input_mask=inp_mask)
        scale = tf.math.sigmoid(scale)

        return loc, scale


class AttnLayer(tf.Module):
    def __init__(self, config, dtype, name="attn_layer", is_training=True):
        super().__init__(name=name)
        self.initializer = tf.keras.initializers.TruncatedNormal(
            mean=0.0, stddev=0.05, seed=None)
        self.config = config
        self.dtype = dtype
        self.is_training = is_training

    @tf.Module.with_name_scope
    def __call__(self, inp, inp_mask=None, bi_causal_mask=False):
        config = self.config
        inputs = tf.transpose(inp, [1, 0, 2])
        qlen = tf.shape(inputs)[0]

        if inp_mask is not None:
            input_mask = tf.transpose(inp_mask)
            input_mask = input_mask[None, :, :, None]

            attn_mask = tf.cast(input_mask > 0, dtype=self.dtype)
        else:
            attn_mask = None

        if bi_causal_mask:
            fw_causal_mask = modeling._create_mask(qlen, 0, self.dtype, False)
            fw_causal_mask = fw_causal_mask[:, :, None, None]

            bw_causal_mask = modeling.attention_mask_ops(qlen, qlen, self.dtype)
            bw_causal_mask = bw_causal_mask[:, :, None, None]

            if attn_mask is not None:
                fw_causal_mask += attn_mask
                fw_attn_mask = tf.cast(fw_causal_mask > 0, dtype=self.dtype)
                bw_causal_mask += attn_mask
                bw_attn_mask = tf.cast(bw_causal_mask > 0, dtype=self.dtype)
            else:
                fw_attn_mask = fw_causal_mask
                bw_attn_mask = bw_causal_mask

        output_h = inputs

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

        if bi_causal_mask:
            fw_output_h = output_h
            bw_output_h = output_h
            for attn, ffn in zip(self.multi_attns, self.post_ffns):
                fw_output_h = attn(fw_output_h, fw_output_h, fw_output_h, fw_attn_mask)
                fw_output_h = ffn(fw_output_h)
                bw_output_h = attn(bw_output_h, bw_output_h, bw_output_h, bw_attn_mask)
                bw_output_h = ffn(bw_output_h)
        else:
            for attn, ffn in zip(self.multi_attns, self.post_ffns):
                output_h = attn(output_h, output_h, output_h, attn_mask)
                output_h = ffn(output_h)

        if config.pre_ln:
            if not hasattr(self, "layer_norm"):
                self.layer_norm = tf.keras.layers.LayerNormalization(name='LayerNorm_out')

            if bi_causal_mask:
                fw_output_h = self.layer_norm(fw_output_h)
                bw_output_h = self.layer_norm(bw_output_h)

                return fw_output_h, bw_output_h
            else:
                output_h = self.layer_norm(output_h)
                return output_h


class NN(tf.Module):
    def __init__(self, layer_num_list, out_dim, dtype, name="NN"):
        super().__init__(name=name)
        self.dtype = dtype
        self.initializer = tf.keras.initializers.TruncatedNormal(
            mean=0.0, stddev=0.05, seed=None)
        self.layer_num_list = layer_num_list
        self.out_dim = out_dim

    @tf.Module.with_name_scope
    def __call__(self, inp):
        t_out = inp
        s_out = inp

        if not hasattr(self, "layer_list"):
            self.layer_list = []
            for i, size in enumerate(self.layer_num_list):
                s_dense = tf.keras.layers.Dense(
                    size, activation=tf.nn.relu, dtype=self.dtype,
                    kernel_initializer=self.initializer, name="s_{}".format(i))
                t_dense = tf.keras.layers.Dense(
                    size, activation=tf.nn.relu, dtype=self.dtype,
                    kernel_initializer=self.initializer, name="t_{}".format(i))

                self.layer_list.append((s_dense, t_dense))

            self.t_proj = tf.keras.layers.Dense(
                    self.out_dim, activation=None, dtype=self.dtype,
                    kernel_initializer=self.initializer, name="t_proj")
            self.s_proj = tf.keras.layers.Dense(
                    self.out_dim, activation=tf.nn.tanh, dtype=self.dtype,
                    kernel_initializer=self.initializer, name="log_s_layer")

        for s_d, t_d in self.layer_list:
            t_out = t_d(t_out)
            s_out = s_d(s_out)
        t_out = self.t_proj(t_out)
        log_s = self.s_proj(s_out)

        return log_s, t_out


class RealNVP2(tf.Module):
    def __init__(self, num_realnvp, output_dim, hidden_layers, dtype=tf.float32,
                 name="realnvp"):
        assert output_dim % 2 == 0
        super().__init__(name=name)
        self.output_dim = output_dim
        self.num_realnvp = num_realnvp
        self.dtype = dtype
        self.hidden_layers = hidden_layers

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
    def __call__(self, loc, scale, direction="forward", sample_size=1, inp=None):
        if not hasattr(self, "nn_layers"):
            self.nn_layers = []
            for i in range(self.num_realnvp):
                nn = NN(self.hidden_layers,
                        self.output_dim // 2,
                        name="nn_{}".format(i),
                        dtype=self.dtype)
                self.nn_layers.append(nn)

        if direction == "forward":
            direction = 1
            inp = self._norm_diag_sample(sample_size, loc, scale)
        elif direction == "invert":
            direction = -1
        else:
            raise ValueError("Arg direction should be 'forward' or 'invert'.")

        log_det_inv = 0
        x = inp

        for i in range(self.num_realnvp)[::direction]:
            x_a, x_b = tf.split(x, 2, axis=-1)
            if i % 2 == 0:
                resid = x_a
                trans = x_b
            else:
                resid = x_b
                trans = x_a

            log_s, t = self.nn_layers[i](resid)
            log_det_inv += log_s
            s = tf.exp(log_s)
            if direction == 1:
                y_trans = s * trans + t
            else:
                y_trans = (trans - t) / s

            if i % 2 == 0:
                y = tf.concat([resid, y_trans], axis=-1)
            else:
                y = tf.concat([y_trans, resid], axis=-1)

            x = y

        return x, log_det_inv


class RealNVP(tfp.bijectors.Bijector):
    def __init__(self,
                 output_dim,
                 hidden_layers,
                 # this bijector do vector wise quantities.
                 forward_min_event_ndims=1,
                 validate_args=False,
                 name="realnvp"):
        super(RealNVP, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name
        )

        assert output_dim % 2 == 0
        self.output_dim = output_dim
        nn_layer = NN(hidden_layers, self.output_dim // 2)
        self.nn = nn_layer

    def _forward(self, x):
        x_a, x_b = tf.split(x, 2, axis=-1)
        y_b = x_b
        log_s, t = self.nn(x_b)
        s = tf.exp(log_s)
        y_a = s * x_a + t

        y = tf.concat([y_a, y_b], axis=-1)
        return y

    def _inverse(self, y):
        y_a, y_b = tf.split(y, 2, axis=-1)
        x_b = y_b
        log_s, t = self.nn(y_b)
        s = tf.exp(log_s)
        x_a = (y_a - t) / s
        x = tf.concat([x_a, x_b], axis=-1)
        return x

    def _forward_log_det_jacobian(self, x):
        _, x_b = tf.split(x, 2, axis=-1)
        log_s, t = self.nn(x_b)
        return log_s
