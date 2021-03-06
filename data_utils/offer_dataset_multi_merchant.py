import tensorflow as tf

from data_utils.preprocess import preprocess_token


def _add_pos_pair_and_label(example):
    def _duplicate_first(tensor):
        dup_tensor = tf.concat([tensor, tensor], axis=0)
        return dup_tensor

    example["combined_padded"] = _duplicate_first(example["combined_padded"])
    example["masked_inp_padded"] = _duplicate_first(example["masked_inp_padded"])
    example["cate_pos_padded"] = _duplicate_first(example["cate_pos_padded"])
    example["mlm_pos_padded"] = _duplicate_first(example["mlm_pos_padded"])
    example["attn_mask"] = _duplicate_first(example["attn_mask"])
    example["cate_str_id"] = _duplicate_first(example["cate_str_id"])
    example["cate_l1_id"] = _duplicate_first(example["cate_l1_id"])

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
    example["cate_str_id"] = tf.reshape(example["cate_str_id"], [-1])
    example["cate_l1_id"] = tf.reshape(example["cate_l1_id"], [-1])
    if "cate_l1" in example:
        example["cate_l1"] = tf.reshape(example["cate_l1"], [-1])
    if "cate_l2" in example:
        example["cate_l2"] = tf.reshape(example["cate_l2"], [-1])

    pair_idx = example["pos_pair_idx"]

    range_tensor = tf.range(0, batch_size, delta=4, dtype=tf.int32)
    range_tensor_t = tf.transpose(range_tensor[None])

    tile_tensor = tf.tile(range_tensor_t, [1, 4])

    shift_idx_tensor = tf.reshape(tile_tensor, [batch_size//2, 2])

    pair_idx = tf.reshape(pair_idx, [batch_size//2, 2])
    pair_idx = pair_idx + shift_idx_tensor

    example["pos_pair_idx"] = pair_idx

    return example


def batch_neg_pair(ds, total_batch_size, inp_batch=4):
    assert total_batch_size % inp_batch == 0
    batch = total_batch_size // inp_batch

    ds = (
        ds.batch(batch, drop_remainder=True)
        .map(_explode_batch_and_shift_pair_idx)
    )

    return ds


def create_neg_pair_dataset(
    ds_list, no_cate_ds, batch_size, inp_len=256, BOS_id=50000, EOS_id=50001, SEP_id=50002,
    PAD_id=50001, MSK_id=50003, mask_prob=0.15, rand_token_size=50000, add_cate_prob=0.1,
    add_mlm_token=True, prefetch=256, remove_non_use=True
):

    def _check_catel2_equal(example):
        cate_l2 = example["cate_l2"]
        rev_l2 = tf.reverse(cate_l2, axis=[-1])
        not_eq = tf.math.not_equal(cate_l2, rev_l2)
        all_not_eq = tf.reduce_all(not_eq)
        return all_not_eq

    def _remove_non_use_tensor(example):
        del example["cate_l1"]
        del example["cate_l2"]

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

    def _explode_zip_v2(pair_exam, no_pair_exam):
        keys = list(pair_exam.keys())
        combined_example = {}
        for key in keys:
            v = []
            v.append(pair_exam[key])

            if key in no_pair_exam:
                v.append(no_pair_exam[key])
                batch_v = tf.concat(v, axis=0)
            else:
                batch_v = pair_exam[key]
            combined_example[key] = batch_v

        return combined_example

    def _process_ds(ds, remove_non_use):
        ds = preprocess_token(ds, inp_len=inp_len, BOS_id=BOS_id, EOS_id=EOS_id, SEP_id=SEP_id,
                              PAD_id=PAD_id, MSK_id=MSK_id, mask_prob=mask_prob,
                              rand_token_size=rand_token_size, add_cate_prob=add_cate_prob,
                              add_mlm_token=add_mlm_token)

        ds3 = ds.shuffle(batch_size*32)
        ds4 = ds3.group_by_window(lambda x: x["cate_l1_id"],
                                  lambda key, ds: ds.batch(2, drop_remainder=True), window_size=2)
        ds5 = ds4.filter(_check_catel2_equal)
        # bsz: 4
        ds6 = ds5.map(_add_pos_pair_and_label)
        if remove_non_use:
            ds6 = ds6.map(_remove_non_use_tensor)

        return ds6

    def _process_no_cate_ds(ds, bsz, remove_non_use):
        ds = preprocess_token(ds, inp_len=inp_len, BOS_id=BOS_id, EOS_id=EOS_id, SEP_id=SEP_id,
                              PAD_id=PAD_id, MSK_id=MSK_id, mask_prob=mask_prob,
                              rand_token_size=rand_token_size, add_cate_prob=add_cate_prob,
                              add_mlm_token=add_mlm_token)

        # ds2 = ds.filter(_filter_cate_length)
        ds3 = ds.shuffle(batch_size*32)
        if remove_non_use:
            ds3 = ds3.map(_remove_non_use_tensor)
        ds4 = ds3.batch(bsz, drop_remainder=True)
        return ds4

    num_ds = len(ds_list)
    proced_ds = tuple([_process_ds(ds, remove_non_use) for ds in ds_list])
    zip_ds = tf.data.Dataset.zip(tuple(proced_ds))
    zip_batch_ds = zip_ds.map(_explode_zip)
    pre_batch = num_ds * 4
    if no_cate_ds is not None:
        left_batch = batch_size % pre_batch
        # batch_size-4 4 for no_cate_proced_ds
        ds7 = batch_neg_pair(zip_batch_ds, batch_size - left_batch, inp_batch=pre_batch)

        no_cate_proced_ds = _process_no_cate_ds(no_cate_ds, left_batch, remove_non_use)
        zip_ds2 = tf.data.Dataset.zip(tuple([ds7, no_cate_proced_ds]))
        zip_batch_ds2 = zip_ds2.map(_explode_zip_v2)
    else:
        zip_batch_ds2 = batch_neg_pair(zip_batch_ds, batch_size, inp_batch=pre_batch)

    ds8 = zip_batch_ds2.prefetch(prefetch)

    return ds8
