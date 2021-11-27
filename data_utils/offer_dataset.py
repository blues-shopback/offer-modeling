import functools

import tensorflow as tf


feature_description = {
    # 'offer_id': tf.io.FixedLenFeature([], tf.string),
    'cate_l1': tf.io.FixedLenFeature([], tf.string),
    'cate_l2': tf.io.FixedLenFeature([], tf.string),
    'title_enc': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    'x_desc1_enc': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
}


def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)


def _combine_inp(example_proto, inp_len=256, BOS_id=50000, EOS_id=50001, SEP_id=50002):
    title_enc = example_proto["title_enc"]
    desc_enc = example_proto["x_desc1_enc"]

    combined = tf.concat(
        [[BOS_id], title_enc, [SEP_id], desc_enc, [EOS_id]],
        axis=0
    )[:inp_len]

    cate_pos_token = tf.concat(
        [[0], tf.zeros_like(title_enc), [0], tf.ones_like(desc_enc), [1]],
        axis=0
    )[:inp_len]

    combined = tf.cast(combined, tf.int32)

    example_proto["combined"] = combined
    example_proto["cate_pos"] = cate_pos_token

    return example_proto


def _MLM_token(example_proto, MSK_id=50003, prob=0.15, rand_token_size=50000):
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
    rand_token_mask1 = tf.where(tf.math.logical_and((rand2 >= 0.1), (rand2 < 0.2)), x=1.0, y=0.)
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


def _pad_inp(example_proto, inp_len=256, PAD_id=50001):
    combined = example_proto["combined"]
    masked_inp = example_proto["masked_inp"]
    mlm_pos = example_proto["mlm_pos"]
    cate_pos = example_proto["cate_pos"]

    pad_len = inp_len - tf.shape(combined)[0]
    combined_padded = tf.pad(combined, [[0, pad_len]], constant_values=PAD_id)
    masked_inp_padded = tf.pad(masked_inp, [[0, pad_len]], constant_values=PAD_id)
    cate_pos_padded = tf.pad(cate_pos, [[0, pad_len]], constant_values=2)
    # -1 for not eval position.
    mlm_pos_padded = tf.pad(mlm_pos, [[0, pad_len]], constant_values=-1)

    example_proto["combined_padded"] = combined_padded
    example_proto["masked_inp_padded"] = masked_inp_padded
    example_proto["mlm_pos_padded"] = mlm_pos_padded
    example_proto["cate_pos_padded"] = cate_pos_padded

    attn_mask = tf.concat(
        [tf.zeros_like(combined, dtype=tf.float32), tf.ones(pad_len, dtype=tf.float32)],
        axis=0
    )
    example_proto["attn_mask"] = attn_mask

    return example_proto


def _select_tensor(example_proto):
    del example_proto["title_enc"]
    del example_proto["x_desc1_enc"]
    del example_proto["combined"]
    del example_proto["masked_inp"]
    del example_proto["mlm_pos"]
    del example_proto["cate_pos"]

    return example_proto


def _add_pos_pair_and_label(example):
    def _duplicate_first(tensor):
        dup_tensor = tf.concat([tensor, tensor], axis=0)
        return dup_tensor

    example["combined_padded"] = _duplicate_first(example["combined_padded"])
    example["masked_inp_padded"] = _duplicate_first(example["masked_inp_padded"])
    example["cate_pos_padded"] = _duplicate_first(example["cate_pos_padded"])
    example["mlm_pos_padded"] = _duplicate_first(example["mlm_pos_padded"])
    example["attn_mask"] = _duplicate_first(example["attn_mask"])

    positive_pair1 = tf.constant([0, 2], dtype=tf.int32)
    positive_pair2 = tf.constant([1, 3], dtype=tf.int32)
    # negative_pair1 = tf.constant([0, 1], dtype=tf.int32)
    # negative_pair2 = tf.constant([2, 3], dtype=tf.int32)

    # labels = tf.constant([1, 1, 0, 0], dtype=tf.int32)

    pair_idx = tf.stack([positive_pair1, positive_pair2], axis=0)
    example["pos_pair_idx"] = pair_idx

    return example


def _explode_batch_and_shift_pair_idx(example):
    def _reshape(tensor):
        tensor = tf.reshape(tensor, [-1, tf.shape(tensor)[-1]])
        return tensor

    batch = tf.shape(example["combined_padded"])[0]
    inp_batch = tf.shape(example["combined_padded"])[1]
    batch_size = batch * inp_batch

    example["combined_padded"] = _reshape(example["combined_padded"])
    example["masked_inp_padded"] = _reshape(example["masked_inp_padded"])
    example["cate_pos_padded"] = _reshape(example["cate_pos_padded"])
    example["mlm_pos_padded"] = _reshape(example["mlm_pos_padded"])
    example["attn_mask"] = _reshape(example["attn_mask"])
    # example["pair_labels"] = tf.reshape(example["pair_labels"], [-1, 1])

    pair_idx = example["pos_pair_idx"]

    range_tensor = tf.range(0, batch_size, delta=4, dtype=tf.int32)
    range_tensor_t = tf.transpose(range_tensor[None])

    tile_tensor = tf.tile(range_tensor_t, [1, 4])

    shift_idx_tensor = tf.reshape(tile_tensor, [batch_size//2, 2])

    pair_idx = tf.reshape(pair_idx, [batch_size//2, 2])
    pair_idx = pair_idx + shift_idx_tensor

    example["pos_pair_idx"] = pair_idx

    return example


def preprocess_token(ds, inp_len=256, BOS_id=50000, EOS_id=50001, SEP_id=50002, PAD_id=50001,
                     MSK_id=50003, mask_prob=0.15, rand_token_size=50000):
    combine_inp_fn = functools.partial(
        _combine_inp, inp_len=inp_len, BOS_id=BOS_id, EOS_id=EOS_id, SEP_id=SEP_id)
    MLM_token_fn = functools.partial(
        _MLM_token, MSK_id=MSK_id, prob=mask_prob, rand_token_size=rand_token_size)

    pad_inp_fn = functools.partial(_pad_inp, inp_len=inp_len, PAD_id=EOS_id)
    ds = (
        ds
        .map(_parse_function)
        .map(combine_inp_fn)
        .map(MLM_token_fn)
        .map(pad_inp_fn)
        .map(_select_tensor)
    )
    return ds


def batch_neg_pair(ds, total_batch_size, inp_batch=4):
    assert total_batch_size % inp_batch == 0
    batch = total_batch_size // inp_batch

    ds = (
        ds.batch(batch, drop_remainder=True)
        .map(_explode_batch_and_shift_pair_idx)
    )

    return ds


def create_normal_mlm_dataset(
    ds, batch_size, inp_len=256, BOS_id=50000, EOS_id=50001, SEP_id=50002, PAD_id=50001,
    MSK_id=50003, mask_prob=0.15, rand_token_size=50000
):
    def _remove_non_use_tensor(example):
        del example["cate_l1"]
        del example["cate_l2"]
        return example
    ds = preprocess_token(ds, inp_len=inp_len, BOS_id=BOS_id, EOS_id=EOS_id, SEP_id=SEP_id,
                          PAD_id=PAD_id, MSK_id=MSK_id, mask_prob=mask_prob,
                          rand_token_size=rand_token_size)
    ds = (
        ds
        .map(_remove_non_use_tensor)
        .batch(batch_size)
    )

    return ds


def create_neg_pair_dataset(
    ds, batch_size, inp_len=256, BOS_id=50000, EOS_id=50001, SEP_id=50002, PAD_id=50001,
    MSK_id=50003, mask_prob=0.15, rand_token_size=50000
):

    def _hash_cate_l1(example, num_buckets=1569):
        cate_l1 = tf.strings.to_hash_bucket(example["cate_l1"], num_buckets=num_buckets)
        example["l1_hash"] = cate_l1
        return example

    def _check_catel2_equal(example):
        cate_l2 = example["cate_l2"]
        rev_l2 = tf.reverse(cate_l2, axis=[-1])
        not_eq = tf.math.not_equal(cate_l2, rev_l2)
        all_not_eq = tf.reduce_all(not_eq)
        return all_not_eq

    def _remove_non_use_tensor(example):
        del example["cate_l1"]
        del example["cate_l2"]
        del example["l1_hash"]
        return example

    ds = preprocess_token(ds, inp_len=inp_len, BOS_id=BOS_id, EOS_id=EOS_id, SEP_id=SEP_id,
                          PAD_id=PAD_id, MSK_id=MSK_id, mask_prob=mask_prob,
                          rand_token_size=rand_token_size)

    ds2 = ds.map(_hash_cate_l1)
    ds3 = ds2.shuffle(batch_size*16)
    ds4 = ds3.group_by_window(lambda x: x["l1_hash"], lambda key, ds: ds.batch(2), window_size=2)
    ds5 = ds4.filter(_check_catel2_equal)

    # bsz: 4
    ds6 = ds5.map(_add_pos_pair_and_label).map(_remove_non_use_tensor)
    ds7 = batch_neg_pair(ds6, batch_size)

    return ds7


def create_neg_pair_dataset_v2(
    ds_list, batch_size, inp_len=256, BOS_id=50000, EOS_id=50001, SEP_id=50002, PAD_id=50001,
    MSK_id=50003, mask_prob=0.15, rand_token_size=50000
):

    def _hash_cate_l1(example, num_buckets=1569):
        cate_l1 = tf.strings.to_hash_bucket(example["cate_l1"], num_buckets=num_buckets)
        example["l1_hash"] = cate_l1
        return example

    def _check_catel2_equal(example):
        cate_l2 = example["cate_l2"]
        rev_l2 = tf.reverse(cate_l2, axis=[-1])
        not_eq = tf.math.not_equal(cate_l2, rev_l2)
        all_not_eq = tf.reduce_all(not_eq)
        return all_not_eq

    def _remove_non_use_tensor(example):
        del example["cate_l1"]
        del example["cate_l2"]
        del example["l1_hash"]
        return example

    def _explode_zip(*example):
        keys = list(example[0].keys())
        combined_example = {}
        for key in keys:
            v = []
            for exa in example:
                v.append(exa[key])
            batch_v = tf.concat(v, axis=0)
            combined_example[key] = batch_v

        return combined_example

    def _process_ds(ds):
        ds = preprocess_token(ds, inp_len=inp_len, BOS_id=BOS_id, EOS_id=EOS_id, SEP_id=SEP_id,
                              PAD_id=PAD_id, MSK_id=MSK_id, mask_prob=mask_prob,
                              rand_token_size=rand_token_size)

        ds2 = ds.map(_hash_cate_l1)
        ds3 = ds2.shuffle(batch_size*16)
        ds4 = ds3.group_by_window(lambda x: x["l1_hash"],
                                  lambda key, ds: ds.batch(2), window_size=2)
        ds5 = ds4.filter(_check_catel2_equal)
        # bsz: 4
        ds6 = ds5.map(_add_pos_pair_and_label).map(_remove_non_use_tensor)
        return ds6

    num_ds = len(ds_list)
    proced_ds = tuple([_process_ds(ds) for ds in ds_list])

    zip_ds = tf.data.Dataset.zip(proced_ds)
    zip_batch_ds = zip_ds.map(_explode_zip)
    pre_batch = num_ds * 4

    ds7 = batch_neg_pair(zip_batch_ds, batch_size, inp_batch=pre_batch)

    return ds7
