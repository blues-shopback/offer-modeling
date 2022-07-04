import os
import argparse

import tensorflow as tf

from utils.offer_model_eval import combine_input
from models.transformer import offer_model
from models.config_utils import BertConfig


def process_model_input(title_enc, cate_enc, desc_enc, max_seq_len):
    combined_pad, cate_pos_token_pad, attn_mask = combine_input(
        title_enc, desc_enc, inp_len=max_seq_len, BOS_id=50000,
        EOS_id=50001, SEP_id=50002, PAD_id=50001, cate_enc=cate_enc)

    return combined_pad, cate_pos_token_pad, attn_mask


def main(args):
    model_path = args.model_path
    max_seq_len = args.max_seq_len
    output_path = args.output_path

    # init model
    config = BertConfig(json_path=os.path.join(model_path, "config.json"))
    config.dropatt = 0.
    config.dropout = 0.
    model = offer_model.OfferModel(config, is_training=False)
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, model_path, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint).expect_partial()

    call = model.traced_call.get_concrete_function(
        tf.TensorSpec([None, max_seq_len], tf.int32),
        tf.TensorSpec([None, max_seq_len], tf.int32),
        tf.TensorSpec([None, max_seq_len], tf.float32),
    )

    tf.saved_model.save(model, output_path, signatures=call)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save model to saved_model format.')
    parser.add_argument('--model_path', type=str,
                        help="Path to model checkpoints.")
    parser.add_argument('--max_seq_len', type=int, default=64,
                        help="Max input token number to process.")
    parser.add_argument('--output_path', type=str,
                        help="Saved model output path.")
    args = parser.parse_args()
    main(args)
