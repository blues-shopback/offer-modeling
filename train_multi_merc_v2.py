import os
import json
import time
import argparse

import numpy as np
import tensorflow as tf

from utils.logging_conf import get_logger

from data_utils.preprocess import preprocess_token
from data_utils.offer_dataset_multi_merchant_v2 import get_dataset

from models.train_utils import CustomSchedule, AdamWeightDecay
from models.transformer import offer_model_v2
from models.config_utils import BertConfig


encoder = None
enc_fn = None


def eval_data(logger, model, args, repr_list, eval_list):
    repr_ds = tf.data.TFRecordDataset(repr_list)
    repr_dataset = preprocess_token(
        repr_ds,
        inp_len=args.max_seq_len,
        BOS_id=50000,
        EOS_id=50001,
        SEP_id=50002,
        PAD_id=50001,
        MSK_id=50003,
        add_mlm_token=False,
        add_cate_prob=1.0
    )
    batch_repr_ds = repr_dataset.batch(args.batch_size)

    eval_ds = tf.data.TFRecordDataset(eval_list)
    eval_dataset = preprocess_token(
        eval_ds,
        inp_len=args.max_seq_len,
        BOS_id=50000,
        EOS_id=50001,
        SEP_id=50002,
        PAD_id=50001,
        MSK_id=50003,
        add_mlm_token=False,
        add_cate_prob=1.0
    )
    batch_eval_ds = eval_dataset.batch(args.batch_size)

    repr_cate_l1_id_tensor_list = []
    repr_output_tensor_list = []

    for example in batch_repr_ds:
        pooled, cate_l1_id = eval_step(model, example)
        repr_output_tensor_list.append(pooled)
        repr_cate_l1_id_tensor_list.append(cate_l1_id)

    # Calculate category repr vector
    repr_cate_l1_id_arr = tf.concat(repr_cate_l1_id_tensor_list, axis=0).numpy()
    repr_output_arr = tf.concat(repr_output_tensor_list, axis=0).numpy()

    idx_to_arr = {}
    for idx, output in zip(repr_cate_l1_id_arr, repr_output_arr):
        if idx in idx_to_arr:
            idx_to_arr[idx].append(output)
        else:
            idx_to_arr[idx] = [output]

    cate_id_list = []
    cate_repr_list = []
    for cate_l1_id in idx_to_arr:
        mean_arr = np.mean(idx_to_arr[cate_l1_id], axis=0)
        cate_repr_list.append(mean_arr)
        cate_id_list.append(cate_l1_id)

    repr_metrix = np.stack(cate_repr_list)

    # Calculate eval offer vector
    eval_cate_l1_id_tensor_list = []
    eval_output_tensor_list = []
    for example in batch_eval_ds:
        pooled, cate_l1_id = eval_step(model, example)
        eval_output_tensor_list.append(pooled)
        eval_cate_l1_id_tensor_list.append(cate_l1_id)

    eval_cate_l1_id_arr = tf.concat(eval_cate_l1_id_tensor_list, axis=0).numpy()
    eval_output_arr = tf.concat(eval_output_tensor_list, axis=0).numpy()

    rank_list = []
    cate_id_arr = np.array(cate_id_list)
    for cate_l1_id, output in zip(eval_cate_l1_id_arr, eval_output_arr):
        cossim = np.einsum("i,ji->j", output, repr_metrix)
        sort_idx = np.argsort(cossim).tolist()[::-1]

        rank_cate_id_list = cate_id_arr[sort_idx].tolist()

        if cate_l1_id in cate_id_list:
            target_rank = rank_cate_id_list.index(cate_l1_id)
            rank_list.append(target_rank)

    # print summary
    total_num = len(rank_list)
    sum_num = sum(rank_list)
    mean = round((sum_num / total_num) + 1, 2)
    total_cate_num = len(cate_id_list)

    logger.info("Eval result: {} / {}".format(mean, total_cate_num))


@tf.function
def eval_step(model, example):
    combined_padded = example["combined_padded"]
    cate_pos_padded = example["cate_pos_padded"]
    attn_mask = example["attn_mask"]
    cate_l1_id = example["cate_l1_id"]

    pooled, output = model(combined_padded, cate_pos_padded, attn_mask)

    return pooled, cate_l1_id


@tf.function
def train_step(model, opt, example, global_step, temperature):
    input = example["output"]
    pos_cate_output = example["pos_cate_output"]
    attn_mask_output = example["attn_mask_output"]
    mlm_pos_output = example["mlm_pos_output"]
    pos_pair_idx = example["pos_pair_idx"]
    contrastive_dedominator_mask = example["contrastive_dedominator_mask"]
    brand_id_output = example["brand_id_output"]
    cate_id_output = example["cate_id_output"]
    merchant_output = example["merchant_output"]

    masked_target = input * mlm_pos_output

    with tf.GradientTape() as tape:
        pooled, output = model(input, pos_cate_output, attn_mask_output)
        batch_mlm_loss = model.get_mlm_loss(output, masked_target)
        batch_contrastive_loss = model.get_contrastive_loss(
            pooled, pos_pair_idx, contrastive_dedominator_mask, temp=temperature)

        mlm_loss = tf.reduce_mean(batch_mlm_loss)
        contrastive_loss = tf.reduce_mean(batch_contrastive_loss)

        # Category
        category_loss = model.get_cate_loss(
            output, attn_mask_output, cate_id_output, merchant_output)

        # Brand
        brand_loss = model.get_brand_loss(
            output, attn_mask_output, brand_id_output
        )

        loss = mlm_loss + contrastive_loss + category_loss + brand_loss

    # train_vars = model.get_trainable_vars()
    train_vars = model.trainable_variables
    grad = tape.gradient(loss, train_vars)
    opt.apply_gradients(zip(grad, train_vars))
    global_step.assign_add(1)
    # gnorm = tf.constant(0., dtype=tf.float32)

    return loss, mlm_loss, contrastive_loss, category_loss, brand_loss


def load_dataset_config(dataset_dir):
    brand_mappings = {}
    brand_path = os.path.join(dataset_dir, "brand_mapping.tsv")
    with open(brand_path, "r") as f:
        for line in f:
            brand, brand_id = line.strip().split("\t")
            brand_mappings[brand] = brand_id

    cate_mappings = {}
    cate_path = os.path.join(dataset_dir, "cate_mapping.txt")
    with open(cate_path, "r") as f:
        for line in f:
            try:
                merchant, cate, cate_id = line.strip().split("@@")
            except ValueError:
                print(line)
                break
            if merchant not in cate_mappings:
                cate_mappings[merchant] = {}
            cate_mappings[merchant][cate] = cate_id

    merchant_cate_num = {}
    cate_num_path = os.path.join(dataset_dir, "merchant_cate_num.txt")
    with open(cate_num_path, "r") as f:
        for line in f:
            merchant, cate_num = line.strip().split(" ")
            merchant_cate_num[merchant] = int(cate_num)

    return brand_mappings, cate_mappings, merchant_cate_num


def init_dataset(args, file_list, prefetch):
    ds_num = args.num_dataset
    batch_size = args.batch_size
    max_seq_len = args.max_seq_len
    ds = get_dataset(
        file_list,
        ds_num,
        batch_size,
        inp_len=max_seq_len,
        BOS_id=50000,
        EOS_id=50001,
        SEP_id=50002,
        MSK_id=50003,
        mask_prob=0.1,
        rand_token_size=50000,
        PAD_id=50001,
        add_mlm_token=True
    )

    ds2 = ds.prefetch(prefetch)

    return ds2


def train(args, logger):
    # save args
    args_dict = vars(args)
    json_path = os.path.join(args.model_dir, "args.json")
    with open(json_path, "w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=True)
    # write args to command format
    command_path = os.path.join(args.model_dir, "args_command.txt")
    with open(command_path, "w") as f:
        arg_str_list = []
        for key, value in args_dict.items():
            arg_str = "--{} {}".format(key, value)
            arg_str_list.append(arg_str)
        f.write(" ".join(arg_str_list))

    # Log tensorboard
    timestamp = str(int(time.time()))
    logdir = os.path.join(args.model_dir, "tensorboard_log")
    file_writer = tf.summary.create_file_writer(
        os.path.join(logdir, "{}_{}".format(args.logfile_name, timestamp))
    )
    file_writer.set_as_default()

    # Init model
    config = BertConfig(json_path=args.config_path)
    config_json_path = os.path.join(args.model_dir, "config.json")
    config.to_json(config_json_path)
    initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
    model = offer_model_v2.OfferModel(config, initializer, is_training=True)
    global_step = tf.Variable(0, name="global_step", dtype=tf.int64)
    learning_rate_schedule = CustomSchedule(
        args.learning_rate, args.min_lr_ratio, args.train_steps, args.warmup_steps)

    # Optimizer
    if args.clip <= 0:
        clip = None
    else:
        clip = args.clip
    if args.weight_decay <= 0:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate_schedule,
            beta_1=args.adam_b1,
            beta_2=args.adam_b2,
            epsilon=args.adam_esp,
            global_clipnorm=clip
        )
    else:
        optimizer = AdamWeightDecay(
            weight_decay_rate=args.weight_decay,
            learning_rate=learning_rate_schedule,
            beta_1=args.adam_b1,
            beta_2=args.adam_b2,
            epsilon=args.adam_esp,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
            global_clipnorm=clip
        )

    # Init checkpoint
    epoch = tf.Variable(1, name="epoch", dtype=tf.int64)
    ckpt = tf.train.Checkpoint(
        step=global_step, optimizer=optimizer, model=model,
        epoch=epoch)
    manager = tf.train.CheckpointManager(ckpt, args.model_dir, max_to_keep=3)

    # Restore checkpoint
    if args.init_checkpoint:
        logger.info("Restore model checkpoint from: {}".format(args.init_checkpoint))
        ckpt.restore(args.init_checkpoint)

    log_str_format = ("Epoch:{:>2} "
                      "| Step {:>7} "
                      "| lr {:>8.2e} "
                      "| loss {:>4.2f} "
                      "| mlm_loss {:>5.2f} "
                      "| cont_loss {:>5.2f} "
                      "| cate_loss {:>5.2f} "
                      "| brand_loss {:>5.2f} "
                      "| time: {:>4.2f}s"
                      "    "
                      )
    # log_eval_format = ("\nEval loss: {:>5.3f}\n"
    #                    "Precision: {:>.3f} | Recall: {:>.3f} | F1: {:>.3f}   ")

    # init dataset
    data_dir = args.train_data_dir
    data_path = args.train_data_path
    file_list = tf.io.gfile.glob(os.path.join(
        data_dir, data_path, "merchant_partition=*", "cate_partition=*", "*.tfrecord"))
    dataset = init_dataset(args, file_list, prefetch=64)

    brand_mappings, cate_mappings, merchant_cate_size = load_dataset_config(data_dir)

    # init classify layer
    logger.info("merchant category size:")
    merchant_category_list = []
    for merchant, cate_size in merchant_cate_size.items():
        merchant_category_list.append((merchant, cate_size + 2))
    logger.info("merchant_category_list: {}".format(merchant_category_list))
    model.build_classify_layer(merchant_category_list)
    model.build_brand_layer(len(brand_mappings) + 2)
    logger.info("brand size: {} + 2".format(len(brand_mappings)))

    # Check model parameters
    exam = next(iter(dataset))
    pooled, output = model(
        exam["output"], pos_cate=exam["pos_cate_output"], inp_mask=exam["attn_mask_output"])
    _ = model.get_cate_loss(
        output, exam["attn_mask_output"], exam["cate_id_output"], exam["merchant_output"])
    _ = model.get_brand_loss(
        output, exam["attn_mask_output"], exam["brand_id_output"])

    tvars = model.trainable_variables
    num_params = sum([np.prod(v.shape) for v in tvars])
    logger.info("Model trainable_variables:")
    for v in tvars:
        logger.info(v.name)
    logger.info('#params: {}'.format(num_params))

    # Variable in loop
    during_t = time.time()
    _t = time.time()
    finished = False
    _loss = 0.
    _mlm_loss = 0.
    _contrastive_loss = 0.
    _cate_loss = 0.
    _brand_loss = 0.
    prev_step = 0
    # Training loop
    logger.info("Training start.")
    dataset_it = iter(dataset)
    while not finished:
        try:
            example = next(dataset_it)
        except StopIteration:
            dataset = init_dataset(args, file_list, prefetch=64)
            dataset_it = iter(dataset)
            example = next(dataset_it)
            epoch.assign_add(1)

        loss, mlm_loss, contrastive_loss, category_loss, brand_loss = train_step(
            model, optimizer, example, global_step, args.temperature)

        # if np.isnan(category_loss.numpy()):
        #     import sys
        #     np.set_printoptions(threshold=sys.maxsize)
        #     print(tf.reduce_sum(example["attn_mask"], axis=1).numpy())
        #     print(loss, mlm_loss, contrastive_loss, category_loss)
        #     break

        # log for tensorboard
        tf.summary.scalar('learning_rate', data=optimizer.lr(global_step), step=global_step)
        tf.summary.scalar('mlm_loss', data=mlm_loss, step=global_step)
        tf.summary.scalar('contrastive_loss', data=contrastive_loss, step=global_step)
        tf.summary.scalar('category_loss', data=category_loss, step=global_step)
        tf.summary.scalar('brand_loss', data=brand_loss, step=global_step)
        tf.summary.scalar('loss', data=loss, step=global_step)

        # print to terminal
        print(log_str_format.format(
            int(epoch),
            int(global_step),
            float(optimizer.lr(global_step)),
            loss, mlm_loss, contrastive_loss,
            category_loss, brand_loss,
            time.time() - _t),
            end="\r")

        # process data for each step
        _t = time.time()
        _loss += loss
        _mlm_loss += mlm_loss
        _contrastive_loss += contrastive_loss
        _cate_loss += category_loss
        _brand_loss += brand_loss

        if int(global_step) > 0 and int(global_step) % args.save_steps == 0:
            save_path = manager.save()
            print()
            logger.info("Save checkpoint to: {}".format(save_path))
            # avg_loss = _loss / (int(global_step) - prev_step)
            # avg_mlm_loss = _mlm_loss / (int(global_step) - prev_step)
            # avg_contract_loss = _contrastive_loss / (int(global_step) - prev_step)
            # avg_amazon_loss = _merchant_loss["amazon"] / merchant_step_counter["amazon"]
            # avg_catch_loss = _merchant_loss["catch"] / merchant_step_counter["catch"]
            # avg_mydeal_loss = _merchant_loss["mydeal"] / merchant_step_counter["mydeal"]
            # log_str = log_str_format.format(
            #     int(epoch_amazon), int(epoch_catch), int(epoch_mydeal),
            #     int(global_step),
            #     float(optimizer.lr(global_step)),
            #     avg_loss,
            #     avg_mlm_loss,
            #     avg_contract_loss,
            #     avg_amazon_loss,
            #     avg_catch_loss,
            #     avg_mydeal_loss
            # )
            # logger.info(log_str)

        if int(global_step) > 0 and int(global_step) % args.print_status == 0:
            avg_loss = _loss / (int(global_step) - prev_step)
            avg_mlm_loss = _mlm_loss / (int(global_step) - prev_step)
            avg_contract_loss = _contrastive_loss / (int(global_step) - prev_step)
            avg_cate_loss = _cate_loss / (int(global_step) - prev_step)
            avg_brand_loss = _brand_loss / (int(global_step) - prev_step)
            log_str = log_str_format.format(
                int(epoch),
                int(global_step),
                float(optimizer.lr(global_step)),
                avg_loss,
                avg_mlm_loss,
                avg_contract_loss,
                avg_cate_loss,
                avg_brand_loss,
                time.time() - during_t
            )
            logger.info(log_str)
            during_t = time.time()
            _loss = 0.
            _mlm_loss = 0.
            _contrastive_loss = 0.
            _cate_loss = 0.
            _brand_loss = 0.
            prev_step = int(global_step)

        if int(global_step) > 0 and int(global_step) % args.eval_steps == 0:
            if args.eval_repr_path and args.eval_data_path:
                print()
                repr_list = tf.io.gfile.glob(os.path.join(args.eval_repr_path, "*.tfrecord"))
                eval_list = tf.io.gfile.glob(os.path.join(args.eval_data_path, "*.tfrecord"))
                eval_data(logger, model, args, repr_list, eval_list)

        if int(global_step) > args.train_steps:
            save_path = manager.save()
            logger.info("Save checkpoint to: {}".format(save_path))
            avg_loss = _loss / (int(global_step) - prev_step)
            avg_mlm_loss = _mlm_loss / (int(global_step) - prev_step)
            avg_contract_loss = _contrastive_loss / (int(global_step) - prev_step)
            avg_cate_loss = _cate_loss / (int(global_step) - prev_step)
            avg_brand_loss = _brand_loss / (int(global_step) - prev_step)
            log_str = log_str_format.format(
                int(epoch),
                int(global_step),
                float(optimizer.lr(global_step)),
                avg_loss,
                avg_mlm_loss,
                avg_contract_loss,
                avg_cate_loss,
                avg_brand_loss,
                time.time() - during_t
            )
            logger.info(log_str)
            logger.info("Finished training.")
            finished = True
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train offer model v2.')
    parser.add_argument('--logfile_name', type=str, default="metrics",
                        help="Tensorboard log name.")
    parser.add_argument('--config_path', type=str, default="", help="Model config path.")
    parser.add_argument('--model_dir', type=str, default="", help="Model save path.")
    parser.add_argument('--train_data_dir', type=str, required=True,
                        help="train data dir with category and brand info inside.")
    parser.add_argument('--train_data_path', type=str, required=True,
                        help="train data dir name inside train_data_dir.")
    parser.add_argument('--eval_steps', type=int, default=10000, help="Steps eval data.")
    parser.add_argument('--eval_repr_path', type=str, default="",
                        help="Path to repr data for eval.")
    parser.add_argument('--eval_data_path', type=str, default="",
                        help="Path to data for eval.")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Model training batch size.")
    parser.add_argument('--max_seq_len', type=int, default=64,
                        help="Maximum sequence length.")
    parser.add_argument('--init_checkpoint', type=str, default=None,
                        help="Model checkpoint to init.")
    parser.add_argument('--train_steps', type=int, default=500000,
                        help="Steps to stop training.")
    parser.add_argument('--gpu', type=str, default="0", help="Gpu to use. ex: 0,1")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--adam_b1', type=float, default=0.9)
    parser.add_argument('--adam_b2', type=float, default=0.99)
    parser.add_argument('--adam_esp', type=float, default=5e-7)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--min_lr_ratio', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip', type=float, default=0., help="Global norm clip value.")
    parser.add_argument('--print_status', type=int, default=1000, help="Steps to print status.")
    # parser.add_argument('--print_exp', type=int, default=1, help="Print example for debug.")
    parser.add_argument('--save_steps', type=int, default=10000, help="Steps to save model.")
    parser.add_argument('--debug', action='store_true', help="Debug mode.")
    parser.add_argument('--num_dataset', type=int, default=16,
                        help="number of file to load to form batch.")

    args = parser.parse_args()

    tf.debugging.set_log_device_placement(args.debug)

    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.model_dir:
        logger = get_logger(working_dir=args.model_dir)
    else:
        logger = get_logger()

    train(args, logger)
