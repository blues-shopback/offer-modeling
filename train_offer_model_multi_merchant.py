import os
import csv
import json
import time
import random
import argparse
import functools
import collections

import numpy as np
import tensorflow as tf

from utils.logging_conf import get_logger
from utils.encoder import get_encoder
from utils.offer_model_eval import eval_query, eval_pairs, encode_and_combine

from data_utils.offer_dataset_multi_merchant import create_neg_pair_dataset

from models.train_utils import CustomSchedule, AdamWeightDecay, CustomWeightSchedule
from models.transformer import offer_model
from models.config_utils import BertConfig, OfferDatasetConfig


encoder = None
enc_fn = None


def eval_step(logger, model, args):
    global encoder, enc_fn
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if encoder is None:
        bpe_path = os.path.join(dir_path, "resources", "bpe", "bpe_merges_20211108.txt")
        encoder = get_encoder(bpe_path, 50000)
    if enc_fn is None:
        enc_fn = functools.partial(
            encode_and_combine,
            encoder=encoder,
            inp_len=args.max_seq_len,
            BOS_id=50000,
            EOS_id=50001,
            SEP_id=50002,
            PAD_id=50001
        )
    # Start eval pairs
    logger.info("Eval pairs")
    pair_file_path = os.path.join(
        dir_path,
        "resources",
        "offer_model_eval_data",
        "pair20211104.csv")

    pairs = []
    with open(pair_file_path, "r") as f:
        reader = csv.DictReader(f)
        for d in reader:
            pairs.append(d)

    pair_result_str = eval_pairs(model, enc_fn, pairs)

    logger.info(pair_result_str)

    # Start eval query
    logger.info("Eval query")
    query_file_path = os.path.join(
        dir_path,
        "resources",
        "offer_model_eval_data",
        "eval_query_iphone_12.csv")
    offers = []
    with open(query_file_path, "r") as f:
        reader = csv.DictReader(f)
        for d in reader:
            offers.append(d)
    query = "iPhone 12"
    query_result_str = eval_query(model, enc_fn, query, offers)
    logger.info(query_result_str)


@tf.function
def train_step(model, opt, example, global_step, mlm_weight_schedule, temperature, merchant):
    combined_padded = example["combined_padded"]
    masked_inp_padded = example["masked_inp_padded"]
    mlm_pos_padded = example["mlm_pos_padded"]
    cate_pos_padded = example["cate_pos_padded"]
    attn_mask = example["attn_mask"]
    pos_pair_idx = example["pos_pair_idx"]
    cate_target = example["cate_id"]

    masked_target = combined_padded * mlm_pos_padded

    with tf.GradientTape() as tape:
        pooled, output = model(masked_inp_padded, cate_pos_padded, attn_mask)
        batch_mlm_loss = model.get_mlm_loss(output, masked_target)
        batch_contrastive_loss = model.get_contrastive_loss(pooled, pos_pair_idx, temp=temperature)

        mlm_loss = tf.reduce_mean(batch_mlm_loss)
        contrastive_loss = tf.reduce_mean(batch_contrastive_loss)

        # Category
        category_loss = model.get_cate_loss(output, attn_mask, cate_target, merchant)

        # category_loss = tf.reduce_mean(batch_category_loss)

        loss = mlm_weight_schedule(global_step) * mlm_loss + contrastive_loss + category_loss

    # train_vars = model.get_trainable_vars(merchant)
    train_vars = model.trainable_variables
    grad = tape.gradient(loss, train_vars)
    opt.apply_gradients(zip(grad, train_vars))
    global_step.assign_add(1)
    # gnorm = tf.constant(0., dtype=tf.float32)

    return loss, mlm_loss, contrastive_loss, category_loss


def get_dataset_file_list(data_path, dataset_config):
    amazon_cate_path = os.path.join(data_path, dataset_config.amazon_cate_path, "id*/*.tfrecord")
    amazon_wo_cate_path = os.path.join(
        data_path, dataset_config.amazon_wo_cate_path, "*.tfrecord")
    catch_cate_path = os.path.join(data_path, dataset_config.catch_cate_path, "id*/*.tfrecord")
    mydeal_cate_path = os.path.join(data_path, dataset_config.mydeal_cate_path, "id*/*.tfrecord")
    mydeal_wo_cate_path = os.path.join(data_path, dataset_config.mydeal_wo_cate_path, "*.tfrecord")

    amazon_cate_file_list = tf.io.gfile.glob(amazon_cate_path)
    amazon_wo_cate_file_list = tf.io.gfile.glob(amazon_wo_cate_path)
    catch_cate_file_list = tf.io.gfile.glob(catch_cate_path)
    mydeal_cate_file_list = tf.io.gfile.glob(mydeal_cate_path)
    mydeal_wo_cate_file_list = tf.io.gfile.glob(mydeal_wo_cate_path)

    return (amazon_cate_file_list, amazon_wo_cate_file_list, catch_cate_file_list,
            mydeal_cate_file_list, mydeal_wo_cate_file_list)


def create_dataset_list(file_list, num_ds):
    random.shuffle(file_list)
    if num_ds <= 1:
        return tf.data.TFRecordDataset(file_list)
    np_file_list = np.array(file_list)

    idx_range = np.arange(len(file_list))
    ran_idx = np.random.permutation(idx_range)

    ds_list = []

    ds_file_num = len(file_list) // num_ds
    for i in range(num_ds):
        start = i * ds_file_num
        ran_idx
        ran_index1 = ran_idx[start:]
        ran_index2 = ran_idx[:start]

        ran_index = np.concatenate((ran_index1, ran_index2))

        ran_file_list = np_file_list[ran_index]
        dataset = tf.data.TFRecordDataset(ran_file_list)
        ds_list.append(dataset)

    return ds_list


def init_dataset(args, file_list, wo_cate_list, prefetch):
    num_ds = args.num_dataset
    if wo_cate_list is not None:
        ds_list = create_dataset_list(file_list, num_ds-1)
        ds_wo_cate = create_dataset_list(wo_cate_list, 1)
    else:
        ds_list = create_dataset_list(file_list, num_ds)
        ds_wo_cate = None
    processed_dataset = create_neg_pair_dataset(
        ds_list,
        ds_wo_cate,
        batch_size=args.batch_size,
        inp_len=args.max_seq_len,
        BOS_id=50000,
        EOS_id=50001,
        SEP_id=50002,
        PAD_id=50001,
        MSK_id=50003,
        mask_prob=0.15,
        rand_token_size=50000,
        add_cate_prob=args.add_cate_prob,
        prefetch=prefetch
    )

    return processed_dataset


def train(args, logger):
    # save args
    args_dict = vars(args)
    json_path = os.path.join(args.model_dir, "args.json")
    with open(json_path, "w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=True)

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
    model = offer_model.OfferModel(config, initializer, is_training=True)
    global_step = tf.Variable(0, name="global_step", dtype=tf.int64)
    learning_rate_schedule = CustomSchedule(
        args.learning_rate, args.min_lr_ratio, args.train_steps, args.warmup_steps)

    mlm_weight_schedule = CustomWeightSchedule(
        start_value=args.mlm_loss_weight_start,
        min_value=args.mlm_loss_weight_min,
        min_steps=args.mlm_loss_weight_min_step
    )

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
    epoch_amazon = tf.Variable(1, name="epoch_amazon", dtype=tf.int64)
    epoch_catch = tf.Variable(1, name="epoch_catch", dtype=tf.int64)
    epoch_mydeal = tf.Variable(1, name="epoch_mydeal", dtype=tf.int64)
    ckpt = tf.train.Checkpoint(
        step=global_step, optimizer=optimizer, model=model,
        epoch_amazon=epoch_amazon, epoch_catch=epoch_catch,
        epoch_mydeal=epoch_mydeal)
    manager = tf.train.CheckpointManager(ckpt, args.model_dir, max_to_keep=3)

    # Restore checkpoint
    if args.init_checkpoint:
        logger.info("Restore model checkpoint from: {}".format(args.init_checkpoint))
        ckpt.restore(args.init_checkpoint)

    log_str_format = ("Epoch: ({:>2}, {:>2}, {:>2}) "
                      "| Step: {:>8} "
                      "| lr {:>9.3e} "
                      "| loss {:>5.3f} "
                      "| mlm_loss {:>5.3f} "
                      "| contrast_loss {:>5.3f} "
                      "| amazon {:>5.3f} "
                      "| catch {:>5.3f} "
                      "| mydeal {:>5.3f} "
                      "    "
                      )
    # log_eval_format = ("\nEval loss: {:>5.3f}\n"
    #                    "Precision: {:>.3f} | Recall: {:>.3f} | F1: {:>.3f}   ")

    # init dataset
    data_path = args.train_data_path
    dataset_config = OfferDatasetConfig(json_path=os.path.join(data_path, "dataset_config.json"))
    logger.info(dataset_config.format_params())
    (amazon_cate_file_list, amazon_wo_cate_file_list, catch_cate_file_list,
     mydeal_cate_file_list, mydeal_wo_cate_file_list) = get_dataset_file_list(
        data_path, dataset_config)
    amazon_df = init_dataset(args, amazon_cate_file_list, amazon_wo_cate_file_list, prefetch=64)
    catch_df = init_dataset(args, catch_cate_file_list, None, prefetch=64)
    mydeal_df = init_dataset(args, mydeal_cate_file_list, mydeal_wo_cate_file_list, prefetch=64)
    dataset_it_map = {
        "amazon": iter(amazon_df),
        "catch": iter(catch_df),
        "mydeal": iter(mydeal_df),
    }
    # init classify layer
    model.build_classify_layer([
        ("amazon", dataset_config.amazon_cate_size),
        ("catch", dataset_config.catch_cate_size),
        ("mydeal", dataset_config.mydeal_cate_size)
    ])

    # Check model parameters
    exam = next(iter(dataset_it_map["amazon"]))
    pooled, output = model(
        exam["masked_inp_padded"], exam["cate_pos_padded"], exam["attn_mask"])
    category_loss = model.get_cate_loss(
        output, exam["attn_mask"],
        exam["cate_id"], "amazon")

    tvars = model.trainable_variables
    num_params = sum([np.prod(v.shape) for v in tvars])
    for v in tvars:
        logger.info(v.name)
    logger.info('#params: {}'.format(num_params))

    # Variable in loop
    finished = False
    _loss = 0.
    _mlm_loss = 0.
    _contrastive_loss = 0.
    prev_step = 0
    merchant_step_counter = collections.Counter({"amazon": 1, "catch": 1, "mydeal": 1})
    _merchant_loss = {
        "amazon": 0.,
        "catch": 0.,
        "mydeal": 0.
    }
    merchant_loss = {
        "amazon": 0.,
        "catch": 0.,
        "mydeal": 0.
    }
    # Training loop
    logger.info("Training start.")
    while not finished:
        # Sample merchant to load data
        merchant = random.choices(["amazon", "catch", "mydeal"], weights=[0., 0.1, 0.])[0]
        merchant_step_counter[merchant] += 1
        try:
            example = next(dataset_it_map[merchant])
        except StopIteration:
            if merchant == "amazon":
                amazon_df = init_dataset(
                    args, amazon_cate_file_list, amazon_wo_cate_file_list, prefetch=128)
                del dataset_it_map[merchant]
                dataset_it_map[merchant] = iter(amazon_df)
                epoch_amazon.assign_add(1)

            elif merchant == "catch":
                catch_df = init_dataset(args, catch_cate_file_list, None, prefetch=64)
                del dataset_it_map[merchant]
                dataset_it_map[merchant] = iter(catch_df)
                epoch_catch.assign_add(1)

            elif merchant == "mydeal":
                mydeal_df = init_dataset(
                    args, mydeal_cate_file_list, mydeal_wo_cate_file_list, prefetch=64)
                del dataset_it_map[merchant]
                dataset_it_map[merchant] = iter(mydeal_df)
                epoch_mydeal.assign_add(1)

            example = next(dataset_it_map[merchant])

        loss, mlm_loss, contrastive_loss, category_loss = train_step(
            model, optimizer, example, global_step, mlm_weight_schedule, args.temperature,
            merchant)

        merchant_loss[merchant] = category_loss

        # log for tensorboard
        tf.summary.scalar('learning rate', data=optimizer.lr(global_step), step=global_step)
        tf.summary.scalar('mlm loss weight', data=mlm_weight_schedule(global_step),
                          step=global_step)
        tf.summary.scalar('mlm_loss', data=mlm_loss, step=global_step)
        tf.summary.scalar('contrastive_loss', data=contrastive_loss, step=global_step)
        tf.summary.scalar('category_loss', data=category_loss, step=global_step)
        tf.summary.scalar('loss', data=loss, step=global_step)

        # print to terminal
        print(log_str_format.format(
            int(epoch_amazon), int(epoch_catch), int(epoch_mydeal),
            int(global_step),
            float(optimizer.lr(global_step)),
            loss, mlm_loss, contrastive_loss,
            merchant_loss["amazon"],
            merchant_loss["catch"],
            merchant_loss["mydeal"]),
            end="\r")

        _loss += loss
        _mlm_loss += mlm_loss
        _contrastive_loss += contrastive_loss
        _merchant_loss[merchant] += merchant_loss[merchant]

        if int(global_step) > 0 and int(global_step) % args.save_steps == 0:
            save_path = manager.save()
            logger.info("Save checkpoint to: {}".format(save_path))
            avg_loss = _loss / (int(global_step) - prev_step)
            avg_mlm_loss = _mlm_loss / (int(global_step) - prev_step)
            avg_contract_loss = _contrastive_loss / (int(global_step) - prev_step)
            avg_amazon_loss = _merchant_loss["amazon"] / merchant_step_counter["amazon"]
            avg_catch_loss = _merchant_loss["catch"] / merchant_step_counter["catch"]
            avg_mydeal_loss = _merchant_loss["mydeal"] / merchant_step_counter["mydeal"]
            log_str = log_str_format.format(
                int(epoch_amazon), int(epoch_catch), int(epoch_mydeal),
                int(global_step),
                float(optimizer.lr(global_step)),
                avg_loss,
                avg_mlm_loss,
                avg_contract_loss,
                avg_amazon_loss,
                avg_catch_loss,
                avg_mydeal_loss
            )
            logger.info(log_str)

        if int(global_step) > 0 and int(global_step) % args.print_status == 0:
            avg_loss = _loss / (int(global_step) - prev_step)
            avg_mlm_loss = _mlm_loss / (int(global_step) - prev_step)
            avg_contract_loss = _contrastive_loss / (int(global_step) - prev_step)
            avg_amazon_loss = _merchant_loss["amazon"] / merchant_step_counter["amazon"]
            avg_catch_loss = _merchant_loss["catch"] / merchant_step_counter["catch"]
            avg_mydeal_loss = _merchant_loss["mydeal"] / merchant_step_counter["mydeal"]
            log_str = log_str_format.format(
                int(epoch_amazon), int(epoch_catch), int(epoch_mydeal),
                int(global_step),
                float(optimizer.lr(global_step)),
                avg_loss,
                avg_mlm_loss,
                avg_contract_loss,
                avg_amazon_loss,
                avg_catch_loss,
                avg_mydeal_loss
            )
            logger.info(log_str)
            _loss = 0.
            _mlm_loss = 0.
            _contrastive_loss = 0.
            prev_step = int(global_step)
            merchant_step_counter = collections.Counter({"amazon": 1, "catch": 1, "mydeal": 1})
            _merchant_loss = {
                "amazon": 0.,
                "catch": 0.,
                "mydeal": 0.
            }

        if int(global_step) > args.train_steps:
            save_path = manager.save()
            logger.info("Save checkpoint to: {}".format(save_path))
            avg_loss = _loss / (int(global_step) - prev_step)
            avg_mlm_loss = _mlm_loss / (int(global_step) - prev_step)
            avg_contract_loss = _contrastive_loss / (int(global_step) - prev_step)
            avg_amazon_loss = _merchant_loss["amazon"] / merchant_step_counter["amazon"]
            avg_catch_loss = _merchant_loss["catch"] / merchant_step_counter["catch"]
            avg_mydeal_loss = _merchant_loss["mydeal"] / merchant_step_counter["mydeal"]
            log_str = log_str_format.format(
                int(epoch_amazon), int(epoch_catch), int(epoch_mydeal),
                int(global_step),
                float(optimizer.lr(global_step)),
                avg_loss,
                avg_mlm_loss,
                avg_contract_loss,
                avg_amazon_loss,
                avg_catch_loss,
                avg_mydeal_loss
            )
            logger.info(log_str)
            logger.info("Finished training.")
            finished = True
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Bert model.')
    parser.add_argument('--logfile_name', type=str, default="metrics",
                        help="Tensorboard log name.")
    parser.add_argument('--config_path', type=str, default="", help="Model config path.")
    parser.add_argument('--model_dir', type=str, default="", help="Model save path.")
    parser.add_argument('--train_data_path', type=str, required=True,
                        help="train data dir with dataset_config.json inside.")
    # parser.add_argument('--eval_data_path', type=str,
    #                     default="/glusterfs/blues/rerank/dataset_20200924/valid_4M.txt",
    #                     help="eval data path.")
    # parser.add_argument('--eval_steps', type=int, default=10000, help="Steps eval data each.")
    # parser.add_argument('--eval_num', type=int, default=100, help="Number of data to eval.")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Model training batch size.")
    parser.add_argument('--max_seq_len', type=int, default=96,
                        help="Maximum sequence length.")
    parser.add_argument('--init_checkpoint', type=str, default=None,
                        help="Model checkpoint to init.")
    parser.add_argument('--train_steps', type=int, default=500000,
                        help="Steps to stop training.")
    parser.add_argument('--gpu', type=str, default="0", help="Gpu to use. ex: 0,1")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--mlm_loss_weight_start', type=float, default=2.0)
    parser.add_argument('--mlm_loss_weight_min', type=float, default=0.1)
    parser.add_argument('--mlm_loss_weight_min_step', type=int, default=1000000)
    parser.add_argument('--adam_b1', type=float, default=0.9)
    parser.add_argument('--adam_b2', type=float, default=0.999)
    parser.add_argument('--adam_esp', type=float, default=1e-7)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--min_lr_ratio', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip', type=float, default=0., help="Global norm clip value.")
    parser.add_argument('--print_status', type=int, default=1000, help="Steps to print status.")
    parser.add_argument('--print_exp', type=int, default=1, help="Print example for debug.")
    parser.add_argument('--save_steps', type=int, default=5000, help="Steps to save model.")
    parser.add_argument('--debug', action='store_true', help="Debug mode.")
    parser.add_argument('--num_dataset', type=int, default=64,
                        help="number of file to load to form batch.")
    parser.add_argument('--add_cate_prob', type=float, default=0.1,
                        help="probibilty for adding category text in input.")

    args = parser.parse_args()

    tf.debugging.set_log_device_placement(args.debug)

    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.model_dir:
        logger = get_logger(working_dir=args.model_dir)
    else:
        logger = get_logger()

    train(args, logger)
