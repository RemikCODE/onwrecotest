import abc

import dill as pickle
import numpy as np
from nltk.lm import Vocabulary
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.backend import ctc_decode

from hwr.constants import DATA, PATH
from hwr.decoding.mlf import label2txt
from hwr.lm.trie import Trie
from hwr.decoding.trie_beam_search import trie_beam_search
from hwr.lm.lm import StupidBackoff, KneserNeyInterpolated, KneserNeyBackoff, MLE


# Interface for a CTC decoding algorithm
class ICTCDecoder:
    def __int__(self):
        __metaclass__ = abc.ABCMeta

    """
    Given the softmax output of RNN of size (batch_size, time_step, labels)
    return top n predictions for each item in batch.
    Output have type list(predictions) of length = batch_size,
    where predictions is list(string) of length = top_n
    """
    @abc.abstractmethod
    def decode(self, rnn_out, top_n, input_lengths=None):
        return


# See trie_beam_search.py
class TrieBeamSearchDecoder(ICTCDecoder):

    def __init__(self, beam_width, lm=None, ngram=0, prune=0, trie=None, gamma=1):
        super().__init__()
        self.beam_width = beam_width
        self.gamma = gamma
        if lm:
            assert ngram
            file_path = PATH.LM_DATA_DIR + str(ngram) + "gram-p" + str(prune) + ".pkl"
            with open(file_path, 'rb') as fin:
                counter = pickle.load(fin)
                vocab = Vocabulary(DATA.CHARS)
            lm_switcher = {
                'mle': MLE(ngram, counter=counter, vocabulary=vocab),
                'sbo': StupidBackoff(ngram, backoff=0.4, counter=counter, vocabulary=vocab),
                'kn': KneserNeyInterpolated(ngram, counter=counter, vocabulary=vocab),
                'knbo': KneserNeyBackoff(ngram, backoff=0.4, counter=counter, vocabulary=vocab),
            }
            lm = lm_switcher[lm]
        self.lm = lm
        self.ngram = ngram
        if trie:
            trie_switcher = {
                '100k': "wiki-100k.txt",
                '10k': 'google-10000-english.txt',
            }
            trie = load_trie(PATH.LM_DATA_DIR + trie_switcher[trie])
        self.trie = trie

    def decode(self, rnn_out, top_n, input_lengths=None):
        if input_lengths is not None:
            rnn_out = [mat[:int(length)] for mat, length in zip(rnn_out, input_lengths)]
        return trie_beam_search(rnn_out, self.beam_width, top_n, gamma=self.gamma,
                                lm=self.lm, lm_order=self.ngram, trie=self.trie)


def load_trie(file_path):
    with open(file_path, encoding='utf8') as fin:
        vocab_txt = fin.read().splitlines()
    # vocabs to lowercase and remove ones with illegal chars
    vocab_txt = [v.lower() for v in vocab_txt if v[:2] != "#!"]
    vocab_txt = [v for v in vocab_txt if all([c in DATA.CHARS for c in v])]
    t = Trie()
    t.mass_insert(vocab_txt)
    return t


class BestPathDecoder(ICTCDecoder):
    def __init__(self):
        super().__init__()

    def decode(self, rnn_out, top_n, input_lengths=None):
        # pred = best_path_tensor(rnn_out)
        pred = best_path(rnn_out, input_lengths=input_lengths)
        return list(map(lambda p: [p for _ in range(top_n)], pred))


class BeamSearchDecoder(ICTCDecoder):
    def __init__(self, beam_width):
        super().__init__()
        self.beam_width = beam_width

    def decode(self, rnn_out, top_n, input_lengths=None):
        return beam_search_tensor(rnn_out, self.beam_width, top_paths=top_n, input_lengths=input_lengths)


# Get max p across all labels at each timestep
def best_path(rnn_out, remove_dup=True, input_lengths=None):
    ret = []
    for i in range(len(rnn_out)):
        length = int(input_lengths[i]) if input_lengths is not None else len(rnn_out[i])
        ret.append([np.argmax(row) for row in rnn_out[i][:length]])
    return label2txt(ret, remove_dup=remove_dup, multiple=True)


# Greedy search. Just pick the most probable candidate at each time step.
def best_path_tensor(rnn_out, input_lengths=None):
    if input_lengths is None:
        input_lengths = np.ones(rnn_out.shape[0]) * rnn_out.shape[1]
    result_list, _ = ctc_decode(rnn_out, input_lengths)
    result_list = result_list[0].numpy()
    pred = label2txt(result_list, multiple=True)
    return list(pred)


# Beam search. Keep track of the best n 'beam' per timestep
# And calculate the prob of them to find the most probable sequence.
def beam_search_tensor(rnn_out, beam_width, top_paths=1, input_lengths=None):
    _EPSILON = 1e-7
    num_of_samples = rnn_out.shape[0]
    if input_lengths is None:
        input_length = np.ones(num_of_samples) * rnn_out.shape[1]
    else:
        input_length = input_lengths
    input_length = tf.convert_to_tensor(input_length, dtype=tf.int32)
    rnn_out = tf.math.log(tf.transpose(rnn_out, perm=[1, 0, 2]) + _EPSILON)

    decoded, _ = tf.compat.v1.nn.ctc_beam_search_decoder(
        inputs=rnn_out,
        sequence_length=input_length,
        beam_width=beam_width,
        top_paths=top_paths,
        merge_repeated=False)

    decoded_dense = [
        tf.sparse.to_dense(st, default_value=-1)
        for st in decoded
    ]
    candidates = [i.numpy() for i in decoded_dense]

    pred = [[] for _ in range(num_of_samples)]
    for k in range(num_of_samples):
        for c in candidates:
            pred[k].append(c[k])
    pred = [list(label2txt(p, multiple=True)) for p in pred]
    return pred
