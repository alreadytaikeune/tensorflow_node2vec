from math import isnan
import tensorflow as tf
import networkx as nx
from scipy.stats import chisquare


mod = tf.load_op_library("./randwalk_ops.so")
vocab, walk, epoch, total, nb_valid = mod.rand_walk_seq(
    "data/miserables.graphml", size=10)
graph = nx.read_graphml("data/miserables.graphml")

walks = []
vocab_to_int = {}
with tf.Session() as sess:
    vocab_, = sess.run([vocab])
    for i, v in enumerate(vocab_):
        vocab_to_int[v] = i
    for i in range(1000):
        walk_, = sess.run([walk])
        walks.append([vocab_[w] for w in walk_])

print("Generated")

stats = {}
exp = {}
for w in walks:
    for i in range(len(w)-1):
        stats.setdefault(w[i], [0]*len(graph))
        exp.setdefault(w[i], [0]*len(graph))
        stats[w[i]][vocab_to_int[w[i+1]]] += 1
        neigh = set(graph.neighbors(w[i]))
        N = len(neigh)
        assert w[i+1] in neigh
        for n in neigh:
            exp[w[i]][vocab_to_int[n]] += 1./N


for w, s in stats.iteritems():
    obs = [s[vocab_to_int[n]] for n in graph.neighbors(w)]
    e = [int(round(exp[w][vocab_to_int[n]])) for n in graph.neighbors(w)]
    print w, obs, e
    _, pvalue = chisquare(obs)
    assert isnan(pvalue) or pvalue > 0.025, pvalue
