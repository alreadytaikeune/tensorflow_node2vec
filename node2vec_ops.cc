/* Copyright 2019 Anis KHLIF. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
  
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/framework/op.h"


REGISTER_OP("Node2VecSeq")
    .Output("node_id: string")
    .Output("walks: int32")
    .Output("nb_seqs_per_node: int32")
    .Output("nb_seqs: int32")
    .Output("nb_valid_nodes: int32")
    .SetIsStateful()
    .Attr("filename: string")
    .Attr("size: int = 40")
    .Attr("p: float = 1")
    .Attr("q: float = 1")
    .Attr("directed: bool = false")
    .Attr("batchsize: int = 128")
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

