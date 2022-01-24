"""Preprocess data from format tf.train.Example"""
import functools

import tensorflow as tf


def _parse_function(example_proto):
    feature_description = {
        'cate_l1': tf.io.FixedLenFeature([], tf.string, default_value=""),
        'cate_l2': tf.io.FixedLenFeature([], tf.string, default_value=""),
        'cate_str': tf.io.FixedLenFeature([], tf.string, default_value=""),
        'title_enc': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'x_desc1_enc': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'x_cate1_enc': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }
    return tf.io.parse_single_example(example_proto, feature_description)


def get_combine_inp_fn(inp_len=256, BOS_id=50000, EOS_id=50001, SEP_id=50002, add_cate_prob=0.1):

    def combine_inp(example_proto, inp_len, BOS_id, EOS_id, SEP_id, add_cate_prob):
        title_enc = example_proto["title_enc"]
        desc_enc = example_proto["x_desc1_enc"]
        cate_enc = example_proto["x_cate1_enc"]

        rand = tf.random.uniform([1], maxval=1.0)

        combined_cate = tf.concat(
            [
                [BOS_id],
                title_enc, [SEP_id], cate_enc, [SEP_id], desc_enc,
                [EOS_id]
            ],
            axis=0
        )[:inp_len]

        combined_wo_cate = tf.concat(
            [
                [BOS_id],
                title_enc, [SEP_id], [SEP_id], desc_enc,
                [EOS_id]
            ],
            axis=0
        )[:inp_len]

        combined = tf.cond(rand < add_cate_prob, lambda: combined_cate, lambda: combined_wo_cate)

        cate_pos_cate = tf.concat(
            [
                [0], tf.zeros_like(title_enc), [0],  # title
                tf.ones_like(cate_enc), [1],  # category
                tf.ones_like(desc_enc) * 2, [2]  # description
            ],
            axis=0
        )[:inp_len]
        cate_pos_wo_cate = tf.concat(
            [
                [0], tf.zeros_like(title_enc), [0],  # title
                [1],  # category
                tf.ones_like(desc_enc) * 2, [2]  # description
            ],
            axis=0
        )[:inp_len]

        cate_pos_token = tf.cond(
            rand < add_cate_prob, lambda: cate_pos_cate, lambda: cate_pos_wo_cate)

        combined = tf.cast(combined, tf.int32)
        cate_pos_token = tf.cast(cate_pos_token, tf.int32)

        example_proto["combined"] = combined
        example_proto["cate_pos"] = cate_pos_token

        return example_proto

    combine_inp_fn = functools.partial(
        combine_inp, inp_len=inp_len, BOS_id=BOS_id, EOS_id=EOS_id, SEP_id=SEP_id,
        add_cate_prob=add_cate_prob)

    return combine_inp_fn


def get_mlm_token_fn(MSK_id=50003, prob=0.15, rand_token_size=50000):

    def mlm_token(example_proto, MSK_id, prob, rand_token_size):
        combined = example_proto["combined"]

        # Assert at least on mask
        ones = tf.zeros(tf.shape(combined)[0] - 1)
        one_mask = tf.concat([[1.], ones], axis=0)
        one_mask = tf.random.shuffle(one_mask)

        # Random mask with probibilty
        # > 0 means masked
        rand = tf.random.uniform(tf.shape(combined), maxval=1.0)
        masks = tf.where(rand < prob, x=1., y=0.)

        # Combine two masks
        mlm_mask = masks + one_mask

        # MLM position
        # masked position will be 1 others will be -1.
        mlm_pos = tf.where(mlm_mask > 0, x=1, y=-1)

        # 10% original token
        rand2 = tf.random.uniform(tf.shape(combined), maxval=1.0)
        ori_mask = tf.where(rand2 < 0.1, x=0., y=1.0)
        mlm_mask = mlm_mask * ori_mask

        # 10% random token
        rand_token_mask1 = tf.where(
            tf.math.logical_and((rand2 >= 0.1), (rand2 < 0.2)), x=1.0, y=0.)
        rand_token_mask = mlm_mask * rand_token_mask1
        rand_tokens = tf.random.uniform(
            tf.shape(combined), minval=0, maxval=rand_token_size, dtype=tf.int32)

        # mask inp
        masks_value = tf.cast(mlm_mask * 1e7, tf.int32)
        masked_inp = tf.where((combined - masks_value) < 0, x=MSK_id, y=combined)

        # mask rand token
        masks_value2 = tf.cast(rand_token_mask * 1e7, tf.int32)
        masked_inp = tf.where((masked_inp - masks_value2) < 0, x=rand_tokens, y=masked_inp)

        example_proto["masked_inp"] = masked_inp
        example_proto["mlm_pos"] = mlm_pos

        return example_proto

    mlm_token_fn = functools.partial(
        mlm_token, MSK_id=50003, prob=0.15, rand_token_size=50000)

    return mlm_token_fn


def get_pad_fn(inp_len=256, PAD_id=50001, add_mlm_token=True):

    def pad_inp(example_proto, inp_len, PAD_id, add_mlm_token):
        combined = example_proto["combined"]
        cate_pos = example_proto["cate_pos"]

        pad_len = inp_len - tf.shape(combined)[0]
        combined_padded = tf.pad(combined, [[0, pad_len]], constant_values=PAD_id)
        cate_pos_padded = tf.pad(cate_pos, [[0, pad_len]], constant_values=3)

        example_proto["combined_padded"] = combined_padded
        example_proto["cate_pos_padded"] = cate_pos_padded

        if add_mlm_token:
            masked_inp = example_proto["masked_inp"]
            mlm_pos = example_proto["mlm_pos"]
            masked_inp_padded = tf.pad(masked_inp, [[0, pad_len]], constant_values=PAD_id)
            # -1 for not eval position.
            mlm_pos_padded = tf.pad(mlm_pos, [[0, pad_len]], constant_values=-1)
            example_proto["masked_inp_padded"] = masked_inp_padded
            example_proto["mlm_pos_padded"] = mlm_pos_padded

        attn_mask = tf.concat(
            [tf.zeros_like(combined, dtype=tf.float32), tf.ones(pad_len, dtype=tf.float32)],
            axis=0
        )
        example_proto["attn_mask"] = attn_mask

        return example_proto

    pad_inp_fn = functools.partial(
        pad_inp, inp_len=inp_len, PAD_id=PAD_id, add_mlm_token=add_mlm_token)

    return pad_inp_fn


def preprocess_token(ds, inp_len=256, BOS_id=50000, EOS_id=50001, SEP_id=50002, PAD_id=50001,
                     MSK_id=50003, mask_prob=0.15, rand_token_size=50000,
                     add_cate_prob=0.1, add_mlm_token=True):

    def _select_tensor(example_proto):
        del example_proto["title_enc"]
        del example_proto["x_desc1_enc"]
        del example_proto["x_cate1_enc"]
        del example_proto["combined"]
        del example_proto["cate_pos"]
        if add_mlm_token:
            del example_proto["masked_inp"]
            del example_proto["mlm_pos"]
        return example_proto

    combine_inp_fn = get_combine_inp_fn(
        inp_len=inp_len, BOS_id=BOS_id, EOS_id=EOS_id, SEP_id=SEP_id, add_cate_prob=add_cate_prob)

    pad_inp_fn = get_pad_fn(inp_len=inp_len, PAD_id=EOS_id, add_mlm_token=add_mlm_token)
    ds1 = (
        ds
        .map(_parse_function)
        .map(combine_inp_fn)
    )
    if add_mlm_token:
        mlm_token_fn = get_mlm_token_fn(
            MSK_id=MSK_id, prob=mask_prob, rand_token_size=rand_token_size)
        ds2 = ds1.map(mlm_token_fn)
    else:
        ds2 = ds1
    ds3 = (
        ds2
        .map(pad_inp_fn)
        .map(_select_tensor)
    )
    return ds3
