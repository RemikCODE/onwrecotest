import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from hwr.constants import PATH, SPLIT
from hwr.data.generator import IAMSequence
from hwr.decoding.ctc_decoder import BestPathDecoder
from hwr.models.ONHWRECO import ONHWRECO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune the pretrained ONHWRECO-LSTM in Keras 3.")
    parser.add_argument("--batch-size", type=int, default=30,
                        help="Training batch size (default: 30, as in the original run).")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Max number of epochs (default: 10).")
    parser.add_argument("--earlystop", type=int, default=5,
                        help="Early stopping patience on val loss (default: 5).")
    parser.add_argument("--preprocess", type=int, default=6,
                        help="Preprocessing scheme (default: 6).")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data dir (default: auto-detected by hwr.constants).")
    parser.add_argument("--split",
                        choices=["train", "test_f", "test_t", "test_v"],
                        default="test_v",
                        help="Split used for watchful evaluation after training.")
    return parser.parse_args()


SPLIT_MAP = {
    "train": SPLIT.TRAIN,
    "test_f": SPLIT.TEST,
    "test_t": SPLIT.VAL2,
    "test_v": SPLIT.VAL1,
}


def main():
    args = parse_args()
    if args.data_dir:
        os.environ["HWR_DATA_DIR"] = args.data_dir

    print("Data dir : {}".format(PATH.DATA_DIR))
    print("Model dir: {}".format(PATH.MODEL_DIR))

    # Load the architecture with the original CuDNNLSTM weights, converted to
    # the Keras 3 LSTM graph at load time (see hwr.models.model._load_legacy_weights).
    print("Loading ONHWRECO-LSTM with pretrained weights ...")
    model = ONHWRECO(preload=True, gru=False, decoder=BestPathDecoder())

    print("Preparing train/val sequences (preprocess={}) ...".format(args.preprocess))
    train_seq = IAMSequence(split=SPLIT.TRAIN, batch_size=args.batch_size,
                            preprocess=args.preprocess)
    val_seq = IAMSequence(split=SPLIT.VAL1, batch_size=args.batch_size,
                          preprocess=args.preprocess)
    print("Train samples: {}, Val samples: {}".format(train_seq.n, val_seq.n))
    if train_seq.n < 10:
        raise RuntimeError(
            "Too few training samples ({}) - is the IAM-OnDB dataset present in {}?"
            .format(train_seq.n, PATH.LINE_DATA_DIR))

    ckptdir = model.ckptdir
    print("Checkpoints will be saved to {}".format(ckptdir))
    print("Training ...")
    model.train(train_seq, val_seq, epochs=args.epochs, earlystop=args.earlystop)

    # Watchful evaluation on the requested split with the freshly trained model.
    eval_seq = IAMSequence(split=SPLIT_MAP[args.split],
                           batch_size=args.batch_size, preprocess=args.preprocess,
                           pred=True, pad_to=(900, 225))
    print("Evaluation on split '{}':".format(args.split))
    print(model.evaluate(eval_seq, decoder=BestPathDecoder()))


if __name__ == "__main__":
    main()