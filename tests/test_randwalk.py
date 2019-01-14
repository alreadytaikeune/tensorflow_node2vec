from math import isnan
import tensorflow as tf
import networkx as nx
from tqdm import tqdm
from collections import Counter
from scipy.stats import chisquare


mod = tf.load_op_library("../randwalk_ops.so")

vocab, walk, epoch, total, nb_valid = mod.rand_walk_seq(
    "../data/miserables.graphml", size=10)
graph = nx.read_graphml("../data/miserables.graphml")
vocab_to_int = {}

walks = []
epoch_ = 0
n_epochs = 10
with tf.Session() as sess:
    vocab_, walk_, nb_valid_ = sess.run([vocab, walk, nb_valid])
    vocab_ = [v.decode("utf8") for v in vocab_]
    walks.append([vocab_[w] for w in walk_])
    for i, v in enumerate(vocab_):
        vocab_to_int[v] = i
    print("{} words in the vocab. {} valid nodes".format(
        len(vocab_), nb_valid_))
    assert nb_valid_ == len(graph)
    prev = 0
    with tqdm(total=nb_valid_*n_epochs) as pbar:
        while epoch_ < n_epochs:
            walk_, epoch_, total_ = sess.run([walk, epoch, total])
            pbar.update(total_-prev)
            prev = total_
            walks.append([vocab_[w] for w in walk_])


def test_epoch_walks_per_start_node():
    c = Counter([w[0] for w in walks])
    assert len(c) == len(graph)
    for node in c:
        assert c[node] == n_epochs


def test_distrib():
    stats = {}
    for w in walks:
        for i in range(len(w)-1):
            stats.setdefault(w[i], [0]*len(graph))
            stats[w[i]][vocab_to_int[w[i+1]]] += 1
            neigh = set(graph.neighbors(w[i]))
            assert w[i+1] in neigh

    for w, s in stats.iteritems():
        obs = [s[vocab_to_int[n]] for n in graph.neighbors(w)]
        print(w, obs)
        _, pvalue = chisquare(obs)
        assert isnan(pvalue) or pvalue > 0.025, pvalue


if __name__ == "__main__":
    test_epoch_walks_per_start_node()
    print("Test epoch OK")
    test_distrib()
    print("Test distrib OK")
