from math import isnan
import networkx as nx
from collections import Counter
from scipy.stats import chisquare

from tensorflow_node2vec.utils import generate_random_walks


def test_epoch_walks_per_start_node(walks, graph, n_epochs):
    c = Counter()
    for wbatch in walks:
        c.update(wbatch[:, 0])
    assert len(c) == len(graph)
    for node in c:
        assert c[node] == n_epochs or c[node] == n_epochs+1, c[node]


def test_distrib(graph, vocab, vocab_to_int):
    stats = {}
    for wbatch in walks:
        for i in range(wbatch.shape[0]):
            w = wbatch[i]
            for i in range(len(w)-1):
                stats.setdefault(w[i], [0]*len(graph))
                stats[w[i]][w[i+1]] += 1
                neigh = set(graph.neighbors(vocab[w[i]]))
                assert vocab[w[i+1]] in neigh

    for w, s in stats.iteritems():
        obs = [s[vocab_to_int[n]] for n in graph.neighbors(vocab[w])]
        if(len(obs) == 1):
            continue
        print(w, obs)
        _, pvalue = chisquare(obs)
        print(pvalue)
        assert isnan(pvalue) or pvalue > 0.025, pvalue


if __name__ == "__main__":
    n_epochs = 30
    graph = nx.read_graphml("../data/miserables.graphml")
    walks, vocab = generate_random_walks(
        "../data/miserables.graphml", 50, n_epochs, batchsize=10)
    vocab_to_int = dict(zip(vocab, range(len(vocab))))
    test_epoch_walks_per_start_node(walks, graph, n_epochs)
    print("Test epoch OK")
    test_distrib(graph, vocab, vocab_to_int)
    print("Test distrib OK")
