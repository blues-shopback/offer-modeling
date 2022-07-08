"""Preprocess data from format tf.train.Example

    v2 for offer model v2.
"""
import functools

import tensorflow as tf


def _parse_function(example_proto):
    feature_description = {
        'merchant': tf.io.FixedLenFeature([], tf.string, default_value=""),
        'brand_id': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        'cate_id': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        'title_enc': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'x_desc1_enc': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'cate_str_enc': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }
    return tf.io.parse_single_example(example_proto, feature_description)


def get_combine_inp_fn(inp_len=256, BOS_id=50000, EOS_id=50001, SEP_id=50002):

    def combine_inp(example_proto, inp_len, BOS_id, EOS_id, SEP_id):
        title_enc = example_proto["title_enc"]
        desc_enc = example_proto["x_desc1_enc"]
        cate_enc = example_proto["cate_str_enc"]

        desc_rand1 = tf.random.uniform([1], maxval=1.0)
        desc_rand2 = tf.random.uniform([1], maxval=1.0)
        drop_rand = tf.random.uniform([1], maxval=1.0)

        rand_desc_enc1 = tf.cond(
            desc_rand1 < 0.5, lambda: desc_enc, lambda: tf.constant([], dtype=desc_enc.dtype))
        rand_desc_enc2 = tf.cond(
            desc_rand2 < 0.5, lambda: desc_enc, lambda: tf.constant([], dtype=desc_enc.dtype))

        combined_cate = tf.concat(
            [
                [BOS_id],
                title_enc, [SEP_id], cate_enc, [SEP_id], rand_desc_enc1,
                [EOS_id]
            ],
            axis=0
        )[:inp_len]

        combined_wo_cate = tf.concat(
            [
                [BOS_id],
                title_enc, [SEP_id], [SEP_id], rand_desc_enc2,
                [EOS_id]
            ],
            axis=0
        )[:inp_len]

        combined_drop_rand = tf.cond(
                drop_rand < 0.5, lambda: combined_cate, lambda: combined_wo_cate)

        cate_pos_cate = tf.concat(
            [
                [0], tf.zeros_like(title_enc), [0],  # title
                tf.ones_like(cate_enc), [1],  # category
                tf.ones_like(rand_desc_enc1) * 2, [2]  # description
            ],
            axis=0
        )[:inp_len]
        cate_pos_wo_cate = tf.concat(
            [
                [0], tf.zeros_like(title_enc), [0],  # title
                [1],  # category
                tf.ones_like(rand_desc_enc2) * 2, [2]  # description
            ],
            axis=0
        )[:inp_len]

        cate_pos_drop = tf.cond(
            drop_rand < 0.5, lambda: cate_pos_cate, lambda: cate_pos_wo_cate)

        example_proto["combined_cate"] = tf.cast(combined_cate, tf.int32)
        example_proto["combined_wo_cate"] = tf.cast(combined_wo_cate, tf.int32)
        example_proto["combined_drop_rand"] = tf.cast(combined_drop_rand, tf.int32)

        example_proto["cate_pos_cate"] = tf.cast(cate_pos_cate, tf.int32)
        example_proto["cate_pos_wo_cate"] = tf.cast(cate_pos_wo_cate, tf.int32)
        example_proto["cate_pos_drop"] = tf.cast(cate_pos_drop, tf.int32)

        return example_proto

    combine_inp_fn = functools.partial(
        combine_inp, inp_len=inp_len, BOS_id=BOS_id, EOS_id=EOS_id, SEP_id=SEP_id)

    return combine_inp_fn


def get_mlm_token_fn(MSK_id=50003, mask_prob=0.1, rand_token_size=50000):

    def _mlm_token(tensor, MSK_id, mask_prob, rand_token_size):
        # Assert at least on mask
        ones = tf.zeros(tf.shape(tensor)[0] - 1)
        one_mask = tf.concat([[1.], ones], axis=0)
        one_mask = tf.random.shuffle(one_mask)

        # Random mask with probibilty
        # > 0 means masked
        rand = tf.random.uniform(tf.shape(tensor), maxval=1.0)
        masks = tf.where(rand < mask_prob, x=1., y=0.)

        # Combine two masks
        mlm_mask = masks + one_mask

        # MLM position
        # masked position will be 1 others will be -1.
        mlm_pos = tf.where(mlm_mask > 0, x=1, y=-1)

        # 10% original token
        rand2 = tf.random.uniform(tf.shape(tensor), maxval=1.0)
        ori_mask = tf.where(rand2 < 0.1, x=0., y=1.0)
        mlm_mask = mlm_mask * ori_mask

        # 10% random token
        rand_token_mask1 = tf.where(
            tf.math.logical_and((rand2 >= 0.1), (rand2 < 0.2)), x=1.0, y=0.)
        rand_token_mask = mlm_mask * rand_token_mask1
        rand_tokens = tf.random.uniform(
            tf.shape(tensor), minval=0, maxval=rand_token_size, dtype=tf.int32)

        # mask inp
        masks_value = tf.cast(mlm_mask * 1e7, tf.int32)
        masked_inp = tf.where((tensor - masks_value) < 0, x=MSK_id, y=tensor)

        # mask rand token
        masks_value2 = tf.cast(rand_token_mask * 1e7, tf.int32)
        masked_inp = tf.where((masked_inp - masks_value2) < 0, x=rand_tokens, y=masked_inp)

        return masked_inp, mlm_pos

    def mlm_token_fn(example_proto, MSK_id, mask_prob, rand_token_size):

        combined_cate = example_proto["combined_cate"]
        combined_wo_cate = example_proto["combined_wo_cate"]
        combined_drop_rand = example_proto["combined_drop_rand"]

        masked_combined_cate, mlm_pos_combined_cate = _mlm_token(
            combined_cate, MSK_id, mask_prob, rand_token_size)
        masked_combined_wo_cate, mlm_pos_combined_wo_cate = _mlm_token(
            combined_wo_cate, MSK_id, mask_prob, rand_token_size)
        masked_combined_drop_rand, mlm_pos_combined_drop_rand = _mlm_token(
            combined_drop_rand, MSK_id, mask_prob, rand_token_size)

        example_proto["masked_combined_cate"] = masked_combined_cate
        example_proto["mlm_pos_combined_cate"] = mlm_pos_combined_cate
        example_proto["masked_combined_wo_cate"] = masked_combined_wo_cate
        example_proto["mlm_pos_combined_wo_cate"] = mlm_pos_combined_wo_cate
        example_proto["masked_combined_drop_rand"] = masked_combined_drop_rand
        example_proto["mlm_pos_combined_drop_rand"] = mlm_pos_combined_drop_rand

        return example_proto

    ret_fn = functools.partial(
        mlm_token_fn, MSK_id=MSK_id, mask_prob=mask_prob, rand_token_size=rand_token_size)

    return ret_fn


def get_pad_fn(inp_len=64, PAD_id=50001, add_mlm_token=True):

    def _pad_inp(inp, cate_pos, inp_len, PAD_id):
        pad_len = inp_len - tf.shape(inp)[0]
        inp_padded = tf.pad(inp, [[0, pad_len]], constant_values=PAD_id)
        cate_pos_padded = tf.pad(cate_pos, [[0, pad_len]], constant_values=3)
        attn_mask = tf.concat(
            [tf.zeros_like(inp, dtype=tf.float32), tf.ones(pad_len, dtype=tf.float32)],
            axis=0
        )

        return inp_padded, cate_pos_padded, attn_mask

    def _pad_mlm_inp(inp, mlm_pos, inp_len, PAD_id):
        pad_len = inp_len - tf.shape(inp)[0]
        inp_padded = tf.pad(inp, [[0, pad_len]], constant_values=PAD_id)
        mlm_pos_padded = tf.pad(mlm_pos, [[0, pad_len]], constant_values=-1)

        return inp_padded, mlm_pos_padded

    def pad_inp(example_proto, inp_len, PAD_id, add_mlm_token):
        combined_cate = example_proto["combined_cate"]
        combined_wo_cate = example_proto["combined_wo_cate"]
        combined_drop_rand = example_proto["combined_drop_rand"]

        cate_pos_cate = example_proto["cate_pos_cate"]
        cate_pos_wo_cate = example_proto["cate_pos_wo_cate"]
        cate_pos_drop = example_proto["cate_pos_drop"]

        combined_cate_pad, cate_pos_cate_pad, attn_mask = _pad_inp(
            combined_cate, cate_pos_cate, inp_len, PAD_id)
        combined_wo_cate_pad, cate_pos_wo_cate_pad, wo_cate_attn_mask = _pad_inp(
            combined_wo_cate, cate_pos_wo_cate, inp_len, PAD_id)
        combined_drop_pad, cate_pos_drop_pad, drop_attn_mask = _pad_inp(
            combined_drop_rand, cate_pos_drop, inp_len, PAD_id)

        # example_proto["combined_cate_pad"] = combined_cate_pad
        # example_proto["cate_pos_cate_pad"] = cate_pos_cate_pad
        # example_proto["attn_mask"] = attn_mask
        # example_proto["combined_wo_cate_pad"] = combined_wo_cate_pad
        # example_proto["cate_pos_wo_cate_pad"] = cate_pos_wo_cate_pad
        # example_proto["wo_cate_attn_mask"] = wo_cate_attn_mask
        # example_proto["combined_drop_pad"] = combined_drop_pad
        # example_proto["cate_pos_drop_pad"] = cate_pos_drop_pad
        # example_proto["drop_attn_mask"] = drop_attn_mask

        del example_proto["combined_cate"]
        del example_proto["combined_wo_cate"]
        del example_proto["combined_drop_rand"]
        del example_proto["cate_pos_cate"]
        del example_proto["cate_pos_wo_cate"]
        del example_proto["cate_pos_drop"]

        if add_mlm_token:
            masked_combined_cate = example_proto["masked_combined_cate"]
            mlm_pos_combined_cate = example_proto["mlm_pos_combined_cate"]
            masked_combined_wo_cate = example_proto["masked_combined_wo_cate"]
            mlm_pos_combined_wo_cate = example_proto["mlm_pos_combined_wo_cate"]
            masked_combined_drop_rand = example_proto["masked_combined_drop_rand"]
            mlm_pos_combined_drop_rand = example_proto["mlm_pos_combined_drop_rand"]

            masked_combined_cate_pad, mlm_pos_combined_cate_pad = _pad_mlm_inp(
                masked_combined_cate, mlm_pos_combined_cate, inp_len, PAD_id)
            masked_combined_wo_cate_pad, mlm_pos_combined_wo_cate_pad = _pad_mlm_inp(
                masked_combined_wo_cate, mlm_pos_combined_wo_cate, inp_len, PAD_id)
            masked_combined_drop_pad, mlm_pos_combined_drop_pad = _pad_mlm_inp(
                masked_combined_drop_rand, mlm_pos_combined_drop_rand, inp_len, PAD_id)

            # example_proto["masked_combined_cate_pad"] = masked_combined_cate_pad
            # example_proto["mlm_pos_combined_cate_pad"] = mlm_pos_combined_cate_pad
            # example_proto["masked_combined_wo_cate_pad"] = masked_combined_wo_cate_pad
            # example_proto["mlm_pos_combined_wo_cate_pad"] = mlm_pos_combined_wo_cate_pad
            # example_proto["masked_combined_drop_pad"] = masked_combined_drop_pad
            # example_proto["mlm_pos_combined_drop_pad"] = mlm_pos_combined_drop_pad

            del example_proto["masked_combined_cate"]
            del example_proto["mlm_pos_combined_cate"]
            del example_proto["masked_combined_wo_cate"]
            del example_proto["mlm_pos_combined_wo_cate"]
            del example_proto["masked_combined_drop_rand"]
            del example_proto["mlm_pos_combined_drop_rand"]

            output = tf.stack([
                masked_combined_cate_pad,
                masked_combined_wo_cate_pad,
                masked_combined_drop_pad,
                masked_combined_drop_pad
            ], axis=0)

            mlm_pos_output = tf.stack([
                mlm_pos_combined_cate_pad,
                mlm_pos_combined_wo_cate_pad,
                mlm_pos_combined_drop_pad,
                mlm_pos_combined_drop_pad
            ], axis=0)

            example_proto["output"] = output
            example_proto["mlm_pos_output"] = mlm_pos_output

        else:
            output = tf.stack([
                combined_cate_pad,
                combined_wo_cate_pad,
                combined_drop_pad,
                combined_drop_pad
            ], axis=0)

            example_proto["output"] = output

        pos_cate_output = tf.stack([
            cate_pos_cate_pad,
            cate_pos_wo_cate_pad,
            cate_pos_drop_pad,
            cate_pos_drop_pad
        ], axis=0)

        example_proto["pos_cate_output"] = pos_cate_output

        attn_mask_output = tf.stack([
            attn_mask,
            wo_cate_attn_mask,
            drop_attn_mask,
            drop_attn_mask
        ], axis=0)

        example_proto["attn_mask_output"] = attn_mask_output

        return example_proto

    pad_inp_fn = functools.partial(
        pad_inp, inp_len=inp_len, PAD_id=PAD_id, add_mlm_token=add_mlm_token)

    return pad_inp_fn


def add_pos_pair_idx(example):

    positive_pair1 = tf.constant([0, 1], dtype=tf.int32)
    positive_pair2 = tf.constant([2, 3], dtype=tf.int32)

    pair_idx = tf.stack([positive_pair1, positive_pair2], axis=0)
    example["pos_pair_idx"] = pair_idx

    return example


def stack_others(example):
    brand_id = tf.cast(example["brand_id"], tf.int32)
    cate_id = tf.cast(example["cate_id"], tf.int32)
    merchant = example["merchant"]

    stack_brand_id = tf.stack([brand_id, brand_id, brand_id, brand_id], axis=0)
    stack_cate_id = tf.stack([cate_id, cate_id, cate_id, cate_id], axis=0)
    stack_merchant = tf.stack([merchant, merchant, merchant, merchant], axis=0)

    example["brand_id_output"] = stack_brand_id
    example["cate_id_output"] = stack_cate_id
    example["merchant_output"] = stack_merchant

    del example["brand_id"]
    del example["cate_id"]
    del example["merchant"]

    return example


def preprocess(file_list, inp_len=64, BOS_id=50000, EOS_id=50001, SEP_id=50002,
               MSK_id=50003, mask_prob=0.1, rand_token_size=50000,
               PAD_id=50001, add_mlm_token=True):

    combine_inp_fn = get_combine_inp_fn(
        inp_len=inp_len, BOS_id=BOS_id, EOS_id=EOS_id, SEP_id=SEP_id)
    mlm_fn = get_mlm_token_fn(
        MSK_id=MSK_id, mask_prob=mask_prob, rand_token_size=rand_token_size)
    pad_fn = get_pad_fn(inp_len=inp_len, PAD_id=PAD_id, add_mlm_token=add_mlm_token)

    ds = tf.data.TFRecordDataset(file_list)
    ds2 = ds.map(_parse_function)
    ds3 = ds2.map(combine_inp_fn)
    ds4 = ds3.map(mlm_fn)
    ds5 = ds4.map(pad_fn)
    ds6 = ds5.map(add_pos_pair_idx)
    ds7 = ds6.map(stack_others)

    return ds7
