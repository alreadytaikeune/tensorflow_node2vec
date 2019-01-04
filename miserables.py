#! encoding=utf8
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
mod = tf.load_op_library("./node2vec_ops.so")
vocab, walk, epoch, total, nb_valid = mod.node2_vec_seq(
    "data/miserables.graphml", p=1, q=2)

window = 4
walks = []

with tf.Session() as sess:
    vocab_, = sess.run([vocab])
    n = len(vocab_)
    cooc = np.zeros((n, n))
    epoch_ = 0
    while epoch_ < 40:
        walk_, epoch_, total_ = sess.run([walk, epoch, total])
        if total_ % 500 == 0:
            print(" ".join(vocab_[w] for w in walk_))
        walks.append(walk_)

for w in walks:
    for i in range(len(w)):
        for j in range(max(0, i-window), min(len(w), i+window)):
            if i == j:
                continue
            cooc[w[i], w[j]] += 1
            if w[i] != w[j]:
                cooc[w[j], w[i]] += 1

print(np.max(cooc))
cooc /= np.max(cooc)
sns.set()
ax = sns.heatmap(cooc, linewidths=0.5, cbar=False,
                 xticklabels=vocab_, yticklabels=vocab_,
                 cmap="YlGnBu")
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.show()
