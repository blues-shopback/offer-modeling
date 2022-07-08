import functools

import numpy as np
import tensorflow as tf

from data_utils.preprocess_v2 import preprocess


def drop_key(example):
    del example["cate_str_enc"]
    del example["title_enc"]
    del example["x_desc1_enc"]

    return example


def explode_zip(*example):
    keys = list(example[0].keys())
    combined_example = {}
    for key in keys:
        v = []
        for exa in example:
            v.append(exa[key])
        batch_v = tf.concat(v, axis=0)
        combined_example[key] = batch_v

    return combined_example


def flat_batch(example):

    def _reshape(tensor):
        tensor = tf.reshape(tensor, [-1, tf.shape(tensor)[-1]])
        return tensor

    out_batch = tf.shape(example["output"])[0]
    in_batch = tf.shape(example["output"])[1]
    batch_size = out_batch * in_batch

    example["output"] = _reshape(example["output"])
    example["mlm_pos_output"] = _reshape(example["mlm_pos_output"])
    example["pos_cate_output"] = _reshape(example["pos_cate_output"])
    example["attn_mask_output"] = _reshape(example["attn_mask_output"])

    example["brand_id_output"] = tf.reshape(example["brand_id_output"], [-1])
    example["cate_id_output"] = tf.reshape(example["cate_id_output"], [-1])
    example["merchant_output"] = tf.reshape(example["merchant_output"], [-1])

    pair_idx = example["pos_pair_idx"]

    range_tensor = tf.range(0, batch_size, delta=4, dtype=tf.int32)
    range_tensor_t = tf.transpose(range_tensor[None])

    tile_tensor = tf.tile(range_tensor_t, [1, 4])

    shift_idx_tensor = tf.reshape(tile_tensor, [batch_size//2, 2])

    pair_idx = tf.reshape(pair_idx, [batch_size//2, 2])
    pair_idx = pair_idx + shift_idx_tensor

    example["pos_pair_idx"] = pair_idx

    return example


def get_contrastive_dedominator_fn(ds_num, batch_size):

    def _build_mask(ds_num, batch_per_ds):
        batch_per_ds = batch_size // ds_num

        eye = tf.eye(ds_num, ds_num)
        tile_eye = tf.tile(eye[:, :, None], [1, 1, batch_per_ds])
        mask = tf.reshape(tile_eye, [-1, batch_size])
        # // 2 for number of pos pair idx
        batch_mask = tf.tile(mask[:, None, :], [1, batch_per_ds // 2, 1])
        final_mask = tf.reshape(batch_mask, [-1, batch_size])

        return final_mask

    def contrastive_dedominator(example, ds_num, batch_size):
        _ds_num = ds_num
        _batch_size = batch_size

        mask = _build_mask(_ds_num, _batch_size)

        example["contrastive_dedominator_mask"] = mask

        return example

    contrastive_dedominator_fn = functools.partial(
        contrastive_dedominator, ds_num=ds_num, batch_size=batch_size)

    return contrastive_dedominator_fn


def create_multiple_rand_file_list(file_list, num_ds):
    np_file_list = np.array(file_list)

    idx_range = np.arange(len(file_list))
    ran_idx = np.random.permutation(idx_range)

    file_list_list = []

    ds_file_num = len(file_list) // num_ds
    if ds_file_num <= 0:
        ds_file_num = 1
    for i in range(min(num_ds, len(file_list))):
        start = i * ds_file_num
        ran_index1 = ran_idx[start:]
        ran_index2 = ran_idx[:start]

        ran_index = np.concatenate((ran_index1, ran_index2))

        ran_file_list = np_file_list[ran_index]
        file_list_list.append(ran_file_list)

    if ds_file_num <= 1:
        n = num_ds // len(file_list)
        file_list_list = file_list_list + file_list_list * n
        file_list_list = file_list_list[:num_ds]

    return file_list_list


def get_dataset(file_list, ds_num, batch_size,
                inp_len=64, BOS_id=50000, EOS_id=50001, SEP_id=50002,
                MSK_id=50003, mask_prob=0.1, rand_token_size=50000,
                PAD_id=50001, add_mlm_token=True):

    preprocess_fn = functools.partial(
        preprocess, inp_len=inp_len, BOS_id=BOS_id, EOS_id=EOS_id, SEP_id=SEP_id,
        MSK_id=MSK_id, mask_prob=mask_prob, rand_token_size=rand_token_size,
        PAD_id=PAD_id, add_mlm_token=add_mlm_token)

    in_ds_batch = int(batch_size / ds_num / 4)
    ds_list = []
    for rand_file_list in create_multiple_rand_file_list(file_list, ds_num):
        ds = preprocess_fn(rand_file_list)
        ds2 = ds.map(drop_key)
        ds3 = ds2.shuffle(batch_size * 2)
        ds4 = ds3.batch(in_ds_batch, drop_remainder=True)
        ds_list.append(ds4)

    contrastive_dedominator = get_contrastive_dedominator_fn(ds_num, batch_size)

    zip_ds = tf.data.Dataset.zip(tuple(ds_list))
    ds5 = zip_ds.map(explode_zip)
    ds6 = ds5.map(flat_batch)
    ds7 = ds6.map(contrastive_dedominator)

    return ds7
