import abc
import datetime
import os

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence

from hwr.constants import PRETRAINED, DATA, PATH
from hwr.decoding.ctc_decoder import TrieBeamSearchDecoder
from hwr.models.metrics import character_error_rate, word_error_rate
from tqdm import tqdm

# Interface for prediction model
class HWRModel(object):
    def __init__(self, chars=DATA.CHARS, preload_key=None,
                 decoder=None):
        __metaclass__ = abc.ABCMeta
        self.decoder = decoder
        if decoder is None:
            self.decoder = TrieBeamSearchDecoder(beam_width=25, lm="sbo", ngram=7,
                                                 prune=10, trie='100k', gamma=1)
        self.chars = chars
        self.class_name = type(self).__name__
        self.ckptdir = PATH.CKPT_DIR + self.class_name + "/"
        self.char_size = len(chars) + 1
        self.model = self.get_model_conf()
        self.pred_model = self.get_intermediate_model(self.get_prediction_layer())
        self.compile()
        if preload_key:
            self.pretrained = PRETRAINED[preload_key]
            print("preloading model weights from {}".format(self.pretrained))
            self.load_weights(self.pretrained, full_path=True)

    @abc.abstractmethod
    def get_model_conf(self):
        return

    @abc.abstractmethod
    def get_prediction_layer(self):
        return

    @abc.abstractmethod
    def get_input_layer(self):
        return

    @abc.abstractmethod
    def get_optimizer(self):
        return

    @abc.abstractmethod
    def get_loss(self):
        return

    def get_intermediate_model(self, layer_name):
        in_model = Model(inputs=self.model.get_layer(self.get_input_layer()).output,
                         outputs=self.model.get_layer(layer_name).output)
        # dummy loss and optimizer, predict with Sequence class requires compiled
        in_model.compile(loss={layer_name: lambda y_true, y_pred: y_pred}, optimizer='adam')
        return in_model


    def train(self, train_seq, test_seq, epochs=100, earlystop=5):
        ckptdir = self.ckptdir + get_time() + '/'
        if not os.path.exists(ckptdir):
            os.makedirs(ckptdir)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(ckptdir + 'weights.weights.h5',
                                                         save_weights_only=True,
                                                         save_best_only=True,
                                                         verbose=1)
        es_callback = tf.keras.callbacks.EarlyStopping(patience=earlystop)
        self.model.fit(
            x=train_seq,
            validation_data=test_seq,
            shuffle=True,
            verbose=1,
            epochs=epochs,
            callbacks=[cp_callback, es_callback]
        )

    def predict_softmax(self, x):
        # for variable length sequence
        if isinstance(x, Sequence) and x.batch_size == 1:
            print("predicting softmax for sequence with batch size: 1, will return list of ndarray.")
            sm = []
            gen = x.gen_iter()
            for b in tqdm(gen, total=len(x)):
                sm.append(self.pred_model.predict(b, verbose=0)[0])
        elif isinstance(x, Sequence):
            sm = self.pred_model.predict(x, verbose=1)
        else:
            sm = self.pred_model.predict(x, verbose=1)
        return sm

    # return top n predicted text.
    def predict(self, x, decoder=None, top=1):
        if decoder is None:
            decoder = self.decoder

        softmaxs = self.predict_softmax(x)
        pred = decoder.decode(rnn_out=softmaxs, top_n=top)
        if top == 1:
            try:
                pred = [p[0] for p in pred]
            except IndexError:
                print("Index Error: {}".format(pred))
        return pred

    def evaluate(self, eval_seq, metrics=None, decoder=None):
        if metrics is None:
            metrics = [character_error_rate, word_error_rate]
        if decoder is None:
            decoder = self.decoder
        _, y_true = eval_seq.get_xy()
        y_pred = self.predict(eval_seq, decoder=decoder)
        ret = {}
        for m in metrics:
            ret[m.__name__] = m(y_true, y_pred)
        return ret

    # Keras cannot save custom loss and keras optimizer, so have to recompile after loading
    def compile(self):
        self.model.compile(loss=self.get_loss(),
                           optimizer=self.get_optimizer())

    def save_weights(self, file_name="", full_path=False):
        if not file_name:
            file_name = get_time() + '.h5'
        if not full_path:
            file_name += self.ckptdir
        self.model.save_weights(file_name)

    def load_weights(self, file_name, full_path=False):
        if not full_path:
            file_name += self.ckptdir
        try:
            self.model.load_weights(file_name)
        except (ValueError, TypeError):
            # weights saved by old keras/tf with CuDNNRNN layers
            # have different weight sizes/names than the current topology
            self._load_legacy_weights(file_name)
        self.compile()

    # Old keras (TF 1.x) saved RNN cells as CuDNNLSTM/CuDNNGRU with a combined
    # bias (input + recurrent, e.g. 480 = 2*4*60) while the new keras builds
    # regular LSTM/GRU cells with a smaller bias (240 = 4*60). This loader
    # re-maps the legacy weights onto the current model topology.
    def _load_legacy_weights(self, file_name):
        saved = {}
        with h5py.File(file_name, "r") as f:
            def _visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    parts = name.split("/")
                    prefix = parts[0]
                    wname = parts[-1].split(":")[0]
                    saved.setdefault(prefix, {})[wname] = obj[()]

            f.visititems(_visit)

        saved_by_kind = {}
        for prefix in saved:
            kind = _legacy_layer_kind(prefix)
            if kind is not None:
                saved_by_kind.setdefault(kind, []).append(prefix)

        target_by_kind = {}
        for layer in self.model.layers:
            kind = _legacy_layer_kind(layer.name)
            if kind is None:
                kind = _legacy_layer_kind(layer.__class__.__name__)
            if kind is not None and layer.get_weights():
                target_by_kind.setdefault(kind, []).append(layer)

        for kind, layers in target_by_kind.items():
            src_prefixes = saved_by_kind.get(kind, [])
            if len(src_prefixes) < len(layers):
                raise ValueError(
                    "Cannot convert legacy weights: expected {} layer groups "
                    "for '{}' but found {}.".format(len(layers), kind,
                                                    len(src_prefixes)))
            for idx, layer in enumerate(layers):
                layer.set_weights(_map_legacy_weights(layer, saved[src_prefixes[idx]]))

    def get_model_summary(self):
        return self.model.summary()




# get timestamp
def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


# Classify both the new and the legacy layer names into a small set of kinds
# so that old weights (e.g. `cu_dnnlstm`, `cu_dnnlstm_1` ...) can be matched
# onto the current topology (`lstm`, `lstm_1` ...).
def _legacy_layer_kind(name):
    name = name.lower()
    if "lstm" in name or "gru" in name or "cudnn" in name or "rnn" in name:
        return "rnn"
    if "batch" in name or "batchnorm" in name:
        return "bn"
    if "conv" in name:
        return "conv"
    if "dense" in name:
        return "dense"
    return None


def _map_legacy_weights(layer, src):
    kind = _legacy_layer_kind(layer.name)
    if kind is None:
        kind = _legacy_layer_kind(layer.__class__.__name__)
    order = {
        "rnn": ["kernel", "recurrent_kernel", "bias"],
        "conv": ["kernel", "bias"],
        "dense": ["kernel", "bias"],
        "bn": ["gamma", "beta", "moving_mean", "moving_variance"],
    }.get(kind)
    if order is None:
        raise ValueError(
            "Cannot convert weights for layer '{}'.".format(layer.name))
    weights = []
    for wname in order:
        if wname not in src:
            raise ValueError(
                "Missing weight '{}' for layer '{}'.".format(wname,
                                                             layer.name))
        value = np.asarray(src[wname], dtype="float32")
        if kind == "rnn":
            if wname == "kernel":
                value = _convert_legacy_rnn_kernel(value)
            elif wname == "recurrent_kernel":
                value = _convert_legacy_rnn_recurrent_kernel(value)
            elif wname == "bias":
                value = _convert_legacy_rnn_bias(layer, value)
        if value.shape != layer.get_weights()[len(weights)].shape:
            raise ValueError(
                "Shape mismatch converting layer '{}': got {}, expected {}."
                .format(layer.name, value.shape,
                        layer.get_weights()[len(weights)].shape))
        weights.append(value)
    return weights


def _convert_legacy_rnn_kernel(kernel):
    # Legacy CuDNNLSTM kernel (input_dim, 4*units) stores each gate block in a
    # transposed (flatten/re-fill) layout; match Keras PR #8307 exactly.
    kernels = np.split(np.asarray(kernel, dtype="float32"), 4, axis=1)
    kernels = [ker.reshape(-1).reshape(ker.shape, order="F") for ker in kernels]
    return np.concatenate(kernels, axis=1)


def _convert_legacy_rnn_recurrent_kernel(recurrent_kernel):
    # Each gate block of the recurrent kernel is stored transposed.
    kernels = np.split(np.asarray(recurrent_kernel, dtype="float32"), 4, axis=1)
    kernels = [ker.T for ker in kernels]
    return np.concatenate(kernels, axis=1)


def _convert_legacy_rnn_bias(layer, bias):
    units = int(layer.units)
    flat = np.asarray(bias, dtype="float32").reshape(-1)
    classname = layer.__class__.__name__
    if classname == "LSTM":
        expected = 4 * units
        if flat.size == expected:
            return flat
        # CuDNNLSTM stores input and recurrent bias concatenated
        if flat.size == 2 * expected:
            return flat[:expected] + flat[expected:]
        raise ValueError("Cannot convert LSTM bias of size {}.".format(flat.size))
    if classname == "GRU":
        expected = 3 * units
        if flat.size == expected:
            return np.broadcast_to(flat[np.newaxis, :], (2, expected))
        # CuDNNGRU stores input and recurrent bias concatenated
        if flat.size == 2 * expected:
            return flat.reshape(2, expected)
        raise ValueError("Cannot convert GRU bias of size {}.".format(flat.size))
    raise ValueError("Unsupported RNN layer type: {}.".format(classname))
