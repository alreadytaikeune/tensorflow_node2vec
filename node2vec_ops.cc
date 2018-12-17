#include "tensorflow/core/framework/op.h"


REGISTER_OP("Node2VecSeq")
    .Output("node_id: string")
    .Output("walks: int32")
    .Output("nb_seqs_per_node: int32")
    .Output("nb_seqs: int32")
    .SetIsStateful()
    .Attr("filename: string")
    .Attr("size: int = 40")
    .Attr("p: float = 1")
    .Attr("q: float = 1")
    .Attr("directed: bool = false")
    .Doc(R"doc(
Parses a graph representation in graphml format and produces batches of examples
created using skipgram sampling on walks generated using the node2vec random
walk process.


node_id: A vector of words in the corpus.
walks: The total number of walks produced so far.
nb_seqs_per_node: The minimal number of walks produced so far. This is can be seen as the epoch.
nb_seqs: The total number of sequences that have been generated thus far;
filename: The path of the graphml file containing the graph.
size: The size of the walks to generate.
p: node2vec p parameter.
q: node2vec q parameter.
directed: is the graph directed.
)doc");

