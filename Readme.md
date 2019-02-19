# Presentation

This is the implementation of a tensorflow operation to perform node2vec sequences generation from a graph stored in graphml format.

The operation returns the vocabulary of nodes, a walk, the epoch, the total number of sequences generated up to now, and the number of valid nodes. The vocabulary of nodes is a list of the nodes' id as found in the file representation, and the list index is the node identifier in the walks. A walk is an array of integers representing the nodes. The epoch is the integer division of the number of generated sequences by the number of nodes in the graph. Valid nodes design nodes that have at least one neighbor, and for which we can generate sequences.

## Supported formats and operations

Currently the supported graph input formats are:

- Graphml (the filename must have ".graphml" as extension)
- Edgelist (partial support: a line should look like "node1 node2 [weight]" or should be prefixed by '#' (comments))

Supported sequence generation algorithms are:

- Random walks
- Node2Vec


Directed and undirected graphs are supported. When using graphml, the directed property is read from the file. When using edge list, you should pass the argument `directed=True` to the tensorflow op for the edges to be considered directed.

Weighted edges are supported as well. To indicate that weights should be used, should pass the parameter `has_weights=True` to the tensorflow operation. When using graphml, you should pass the additional parameter `weights_attribute` to indicate which property in the file contains the weight. When using edgelist, the weight should be the third space sperated element of a line. Inconsistencies between arguments and format will throw an error.


## Cooccurrences of the characters for *Les Miserables*

See the script ![miserables.py](miserables.py)
![alt text](data/miserables.png)

# Requirements

This operation depends on the Boost graph library and the Boost filesystem library. You can get it [here](https://www.boost.org/users/history/version_1_67_0.html). You should also have the tensorflow python library installed. It is better if you compile it from sources. If your machine has a fairly common CPU architecture, then you may find a precompiled Python package with CPU optimization [here](https://github.com/lakshayg/tensorflow-build).

# Usage


First you have to compile the op. Just run `make` (this supposes that your boost headers are in `/usr/local/include`. It should generate a file called `libgraphseq_ops.so` that can be loaded by tensorflow in Python.

If you want to support work sharding, you should remove the flag "-DNO_SHARDER=1" in the Makefile.

The boilerplate code to generate sequences looks like this.

```
import tensorflow as tf
mod = tf.load_op_library("/path/to/libgraphseq_ops.so")
vocab, walk, epoch, total, nb_valid = mod.node2_vec_seq("path/to/your/file.graphml", batchsize=256, size=40)

with tf.Session() as sess:
    vocab_, = sess.run([vocab])
    epoch_ = 0
    while epoch_ < 10:
        walk_, epoch_ = sess.run([walk, epoch])

```

Here `walk_` will be a numpy array of size `(256, 40)` containing 256 walks of size 40.


We recommend that you use the functions defined in [utils.py](utils.py) if you intend to use the library as a module. You can also use the script [generate_walks.py](generate_walks.py) to generate sequences to a file. This script will write a file containing sequences, and another containing a vocabulary. The sequences are space separated integers. The integers are the indices of the nodes in the graph internal representation. The correspondance between node ids and nodes is written in a vocabulary file. The node with index i is written at line i. The main reason for that is that node identifiers in the original file can be quite long strings, which would dramatically increase the size of the sequences file, and increase the generation time.


# TODO

- Write **MORE** tests
- Improve memory efficiency (for graphml)
- Give choice as to whether to compute edge aliases