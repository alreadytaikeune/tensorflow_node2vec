import os
import tensorflow as tf
from tqdm import tqdm

this_dir = os.path.dirname(os.path.abspath(__file__))

rw = tf.load_op_library(os.path.join(this_dir, "randwalk_ops.so"))
n2v = tf.load_op_library(os.path.join(this_dir, "node2vec_ops.so"))


def _generate_walks(n_epochs, vocab, walk, epoch, total, nb_valid):
    walks = []
    epoch_ = 0
    with tf.Session() as sess:
        vocab_, walk_, nb_valid_ = sess.run([vocab, walk, nb_valid])
        vocab_ = [v.decode("utf8") for v in vocab_]
        walks.append(walk_)
        prev = 0
        with tqdm(total=nb_valid_*n_epochs) as pbar:
            while epoch_ < n_epochs:
                walk_, epoch_, total_ = sess.run([walk, epoch, total])
                pbar.update(total_-prev)
                prev = total_
                walks.append(walk_)

    return walks, vocab_


def walks_as_words(walks, vocab):
    out = []
    for walk_ in walks:
        out.append([vocab[w] for w in walk_])
    return out


def generate_random_walks(fname, size, epochs, as_words=False):
    vocab, walk, epoch, total, nb_valid = rw.rand_walk_seq(
        fname, size=size)
    walks, vocab_ = _generate_walks(
        epochs, vocab, walk, epoch, total, nb_valid)
    if as_words:
        return walks_as_words(walks, vocab_), vocab_
    return walks, vocab_


def generate_n2v_walks(fname, size, epochs, p=1, q=1, as_words=False):
    vocab, walk, epoch, total, nb_valid = n2v.node2_vec_seq(
        fname, size=size, p=p, q=q)
    walks, vocab_ = _generate_walks(
        epochs, vocab, walk, epoch, total, nb_valid)
    if as_words:
        return walks_as_words(walks, vocab_), vocab_
    return walks, vocab_
