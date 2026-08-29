import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from hwr.constants import PATH, SPLIT
from hwr.data.generator import IAMSequence
from hwr.decoding.ctc_decoder import BestPathDecoder
from hwr.models.ONHWRECO import ONHWRECO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned ONHWRECO checkpoint on an IAM-OnDB split.")
    parser.add_argument("checkpoint", type=str,
                        help="Path to the fine-tuned weights file (e.g. .../weights.weights.h5).")
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--preprocess", type=int, default=6)
    parser.add_argument("--split",
                        choices=["train", "test_f", "test_t", "test_v"],
                        default="test_v")
    return parser.parse_args()


SPLIT_MAP = {
    "train": SPLIT.TRAIN,
    "test_f": SPLIT.TEST,
    "test_t": SPLIT.VAL2,
    "test_v": SPLIT.VAL1,
}


def main():
    args = parse_args()
    model = ONHWRECO(preload=False, gru=False, decoder=BestPathDecoder())
    model.load_weights(args.checkpoint, full_path=True)

    eval_seq = IAMSequence(split=SPLIT_MAP[args.split],
                           batch_size=args.batch_size, preprocess=args.preprocess,
                           pred=True, pad_to=(900, 225))
    print("Evaluating checkpoint '{}' on split '{}' ({} samples)..."
          .format(args.checkpoint, args.split, eval_seq.n))
    print(model.evaluate(eval_seq, decoder=BestPathDecoder()))


if __name__ == "__main__":
    main()
