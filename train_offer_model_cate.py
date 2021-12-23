import os
import csv
import json
import time
import random
import argparse
import functools

import numpy as np
import tensorflow as tf

from utils.logging_conf import get_logger
from utils.encoder import get_encoder
from utils.offer_model_eval import eval_query, eval_pairs, encode_and_combine

from data_utils.offer_dataset import create_neg_pair_dataset_v2

from models.train_utils import CustomSchedule, AdamWeightDecay, CustomWeightSchedule
from models.transformer import offer_model
from models.config_utils import BertConfig


encoder = None
enc_fn = None


def eval_step(logger, model, args):
    global encoder, enc_fn
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if encoder is None:
        bpe_path = os.path.join(dir_path, "resources", "bpe", "bpe_merge.txt")
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
def train_step(model, opt, example, global_step, mlm_weight_schedule, temperature):
    combined_padded = example["combined_padded"]
    masked_inp_padded = example["masked_inp_padded"]
    mlm_pos_padded = example["mlm_pos_padded"]
    cate_pos_padded = example["cate_pos_padded"]
    attn_mask = example["attn_mask"]
    pos_pair_idx = example["pos_pair_idx"]
    cate_target = example["l1_hash"]

    masked_target = combined_padded * mlm_pos_padded

    with tf.GradientTape() as tape:
        pooled, output = model(masked_inp_padded, cate_pos_padded, attn_mask)
        batch_mlm_loss = model.get_mlm_loss(output, masked_target)
        batch_contrastive_loss = model.get_contrastive_loss(pooled, pos_pair_idx, temp=temperature)
        batch_category_loss = model.get_cate_loss(cate_target)
        mlm_loss = tf.reduce_mean(batch_mlm_loss)
        contrastive_loss = tf.reduce_mean(batch_contrastive_loss)
        category_loss = tf.reduce_mean(batch_category_loss)

        loss = mlm_weight_schedule(global_step) * mlm_loss + contrastive_loss + category_loss

    grad = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grad, model.trainable_variables))
    global_step.assign_add(1)
    gnorm = tf.constant(0., dtype=tf.float32)

    return loss, gnorm, mlm_loss, contrastive_loss, category_loss


def create_dataset(file_list, num_ds=16):
    random.shuffle(file_list)

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


def train(args, logger):
    BATCH_SIZE = args.batch_size
    MAX_SEQ_LEN = args.max_seq_len

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

    # Load training data
    file_list = tf.io.gfile.glob(args.train_data_glob_path)

    # Init model
    config = BertConfig(json_path=args.config_path)
    config_json_path = os.path.join(args.model_dir, "config.json")
    config.to_json(config_json_path)
    initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
    model = offer_model.OfferModel(config, initializer, is_training=True, cate_size=args.num_cate)
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
    epoch = tf.Variable(1, name="epoch", dtype=tf.int64)
    ckpt = tf.train.Checkpoint(
        step=global_step, optimizer=optimizer, model=model, epoch=epoch)
    manager = tf.train.CheckpointManager(ckpt, args.model_dir, max_to_keep=3)

    # Restore checkpoint
    if args.init_checkpoint:
        logger.info("Restore model checkpoint from: {}".format(args.init_checkpoint))
        ckpt.restore(args.init_checkpoint)

    _loss = 0.
    _mlm_loss = 0.
    _contrastive_loss = 0.
    _category_loss = 0.
    _gnorm = 0.
    prev_step = 0

    log_str_format = ("Epoch: {:>3} Step: {:>8} | gnorm {:>4.2f} lr {:>9.3e} "
                      "| loss {:>5.3f} "
                      "| mlm_loss {:>5.3f} "
                      "| contrast_loss {:>5.3f} "
                      "| category_loss {:>5.3f} "
                      "    "
                      )
    # log_eval_format = ("\nEval loss: {:>5.3f}\n"
    #                    "Precision: {:>.3f} | Recall: {:>.3f} | F1: {:>.3f}   ")

    _checked = False
    finished = False
    # Training loop
    while not finished:
        # Epoch start
        logger.info("Epoch {} start.".format(int(epoch)))

        # Init dataset
        dataset_list = create_dataset(file_list, num_ds=args.num_dataset)
        processed_dataset = create_neg_pair_dataset_v2(
            dataset_list,
            batch_size=BATCH_SIZE,
            inp_len=MAX_SEQ_LEN,
            BOS_id=50000,
            EOS_id=50001,
            SEP_id=50002,
            PAD_id=50001,
            MSK_id=50003,
            mask_prob=0.15,
            rand_token_size=50000,
            add_cate_prob=args.add_cate_prob
        )

        # Check model parameters
        if not _checked:
            _checked = True
            exam = next(iter(processed_dataset))
            model(exam["masked_inp_padded"], exam["cate_pos_padded"], exam["attn_mask"])
            tvars = model.trainable_variables
            num_params = sum([np.prod(v.shape) for v in tvars])
            for v in tvars:
                logger.info(v.name)
            logger.info('#params: {}'.format(num_params))

        # Epoch loop
        for example in processed_dataset:
            loss, gnorm, mlm_loss, contrastive_loss, category_loss = train_step(
                model, optimizer, example, global_step, mlm_weight_schedule, args.temperature)

            tf.summary.scalar('learning rate', data=optimizer.lr(global_step), step=global_step)
            tf.summary.scalar('mlm loss weight', data=mlm_weight_schedule(global_step),
                              step=global_step)
            tf.summary.scalar('mlm_loss', data=mlm_loss, step=global_step)
            tf.summary.scalar('contrastive_loss', data=contrastive_loss, step=global_step)
            tf.summary.scalar('category_loss', data=category_loss, step=global_step)
            tf.summary.scalar('loss', data=loss, step=global_step)
            tf.summary.scalar('gnorm', data=gnorm, step=global_step)

            print(log_str_format.format(
                int(epoch), int(global_step), gnorm, float(optimizer.lr(global_step)),
                loss, mlm_loss, contrastive_loss, category_loss),
                end="\r")

            _loss += loss
            _mlm_loss += mlm_loss
            _contrastive_loss += contrastive_loss
            _category_loss += category_loss
            _gnorm += gnorm

            if int(global_step) > 0 and int(global_step) % args.eval_steps == 0:
                eval_step(logger, model, args)

            if int(global_step) > 0 and int(global_step) % args.save_steps == 0:
                save_path = manager.save()
                logger.info("Save checkpoint to: {}".format(save_path))
                avg_loss = _loss / (int(global_step) - prev_step)
                avg_mlm_loss = _mlm_loss / (int(global_step) - prev_step)
                avg_contract_loss = _contrastive_loss / (int(global_step) - prev_step)
                avg_category_loss = _category_loss / (int(global_step) - prev_step)
                avg_gnorm = _gnorm / (int(global_step) - prev_step)
                log_str = log_str_format.format(
                    int(epoch), int(global_step), avg_gnorm, float(optimizer.lr(global_step)),
                    avg_loss,
                    avg_mlm_loss,
                    avg_contract_loss,
                    avg_category_loss
                )
                logger.info(log_str)

            if int(global_step) > 0 and int(global_step) % args.print_status == 0:
                avg_loss = _loss / (int(global_step) - prev_step)
                avg_mlm_loss = _mlm_loss / (int(global_step) - prev_step)
                avg_contract_loss = _contrastive_loss / (int(global_step) - prev_step)
                avg_category_loss = _category_loss / (int(global_step) - prev_step)
                avg_gnorm = _gnorm / (int(global_step) - prev_step)
                log_str = log_str_format.format(
                    int(epoch), int(global_step), avg_gnorm, float(optimizer.lr(global_step)),
                    avg_loss,
                    avg_mlm_loss,
                    avg_contract_loss,
                    avg_category_loss
                )
                logger.info(log_str)
                _loss = 0.
                _gnorm = 0.
                _mlm_loss = 0.
                _contrastive_loss = 0.
                _category_loss = 0.
                prev_step = int(global_step)

            if int(global_step) > args.train_steps:
                save_path = manager.save()
                logger.info("Save checkpoint to: {}".format(save_path))
                avg_loss = _loss / (int(global_step) - prev_step)
                avg_gnorm = _gnorm / (int(global_step) - prev_step)
                log_str = log_str_format.format(
                    int(epoch), int(global_step), avg_gnorm, float(optimizer.lr(global_step)),
                    avg_loss,
                    avg_mlm_loss,
                    avg_contract_loss,
                    avg_category_loss
                )
                logger.info(log_str)
                logger.info("Finished training.")
                finished = True
                break

        # Epoch end
        epoch.assign_add(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Bert model.')
    parser.add_argument('--logfile_name', type=str, default="metrics",
                        help="Tensorboard log name.")
    parser.add_argument('--config_path', type=str, default="", help="Model config path.")
    parser.add_argument('--model_dir', type=str, default="", help="Model save path.")
    parser.add_argument('--train_data_glob_path', type=str, required=True,
                        help="train data glob path. ex: /train_data/*.tfrecord")
    # parser.add_argument('--eval_data_path', type=str,
    #                     default="/glusterfs/blues/rerank/dataset_20200924/valid_4M.txt",
    #                     help="eval data path.")
    parser.add_argument('--eval_steps', type=int, default=5000, help="Steps eval data each.")
    # parser.add_argument('--eval_num', type=int, default=100, help="Number of data to eval.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Model training batch size.")
    parser.add_argument('--max_seq_len', type=int, default=256,
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
    parser.add_argument('--num_cate', type=int, default=1569,
                        help="number of category for classified task.")
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
