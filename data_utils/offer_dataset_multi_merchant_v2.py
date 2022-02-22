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
    example["cate_id"] = _duplicate_first(example["cate_id"])
    example["cate_l1_id"] = _duplicate_first(example["cate_l1_id"])

    positive_pair1 = tf.constant([0, 2], dtype=tf.int32)
    positive_pair2 = tf.constant([1, 3], dtype=tf.int32)
    # negative_pair1 = tf.constant([0, 1], dtype=tf.int32)
    # negative_pair2 = tf.constant([2, 3], dtype=tf.int32)

    # labels = tf.constant([1, 1, 0, 0], dtype=tf.int32)

    pair_idx = tf.stack([positive_pair1, positive_pair2], axis=0)
    example["pos_pair_idx"] = pair_idx

    return example


def _shift_pair_idx(example):
    batch_size = tf.shape(example["combined_padded"])[0]

    pair_idx = example["pos_pair_idx"]

    range_tensor = tf.range(0, batch_size, delta=4, dtype=tf.int32)
    range_tensor_t = tf.transpose(range_tensor[None])

    tile_tensor = tf.tile(range_tensor_t, [1, 4])

    shift_idx_tensor = tf.reshape(tile_tensor, [batch_size//2, 2])

    pair_idx = tf.reshape(pair_idx, [batch_size//2, 2])
    pair_idx = pair_idx + shift_idx_tensor

    example["pos_pair_idx"] = pair_idx

    return example


def create_neg_pair_dataset(
    ds_list, batch_size, inp_len=256, BOS_id=50000, EOS_id=50001, SEP_id=50002,
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
        remove_list = ["cate_l1", "cate_l2", "random_idx"]
        for key in remove_list:
            if key in example:
                del example[key]

        return example

    def _explode_random_zip(*example):
        bsz = batch_size
        num_mini_batch = bsz // 4

        # random idx
        total_idx_num = tf.shape(example[0]["combined_padded"])[0] * len(ds_list)
        idx = tf.random.shuffle(tf.range(total_idx_num))
        idxs = tf.slice(idx, [0], [num_mini_batch])

        keys = list(example[0].keys())
        combined_example = {}
        for key in keys:
            v = []
            for exa in example:
                v.append(exa[key])

            # shape: [inp_bsz * num_ds, 4, ...]
            bsz_v = tf.concat(v, axis=0)

            # shape: [id_num, 4, ...]
            select = tf.gather(bsz_v, idxs)

            # shape: [[4, ...] * id_num]
            unpack_sel = tf.unstack(select, axis=0)

            # shape: [4 * id_num, ...]
            batch_v = tf.concat(unpack_sel, axis=0)

            combined_example[key] = batch_v

        combined_example["random_idx"] = idxs

        return combined_example

    def _process_ds(ds, remove_non_use, to_bsz):
        ds = preprocess_token(ds, inp_len=inp_len, BOS_id=BOS_id, EOS_id=EOS_id, SEP_id=SEP_id,
                              PAD_id=PAD_id, MSK_id=MSK_id, mask_prob=mask_prob,
                              rand_token_size=rand_token_size, add_cate_prob=add_cate_prob,
                              add_mlm_token=add_mlm_token)

        ds3 = ds.shuffle(batch_size*64)
        ds4 = ds3.group_by_window(lambda x: x["cate_l1_id"],
                                  lambda key, ds: ds.batch(2, drop_remainder=True), window_size=2)
        ds5 = ds4.filter(_check_catel2_equal)
        # bsz: 4
        ds6 = ds5.map(_add_pos_pair_and_label)
        if remove_non_use:
            ds6 = ds6.map(_remove_non_use_tensor)

        ds7 = ds6.batch(to_bsz, drop_remainder=True)

        return ds7

    num_ds = len(ds_list)
    num_mini_batch = batch_size // 4

    if num_ds > num_mini_batch:
        to_bsz = 1
    elif num_mini_batch % num_ds == 0:
        to_bsz = num_mini_batch // num_ds
    else:
        to_bsz = num_mini_batch // num_ds + 1

    proced_ds = tuple([_process_ds(ds, remove_non_use, to_bsz) for ds in ds_list])
    zip_ds = tf.data.Dataset.zip(proced_ds)
    zip_batch_ds = zip_ds.map(_explode_random_zip)
    zip_batch_ds2 = zip_batch_ds.map(_shift_pair_idx)

    if remove_non_use:
        zip_batch_ds2 = zip_batch_ds2.map(_remove_non_use_tensor)

    ds = zip_batch_ds2.prefetch(prefetch)

    return ds
