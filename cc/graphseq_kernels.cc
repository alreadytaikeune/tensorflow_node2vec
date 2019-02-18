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

#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <iterator>
#include <vector>
#include <cassert>

#include <iostream>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/util/guarded_philox_random.h"


#include <boost/graph/graphml.hpp>
#include <boost/graph/adjacency_list.hpp>

#include "sampling.h"
#include "graph_kernel_base.h"
#include "graphseq_kernels.h"

using namespace tensorflow;

namespace gseq{

Status Node2VecSeqOp::MakeGraphTypeAndInit(Env* env, const string& filename){
    bool has_weight = HasWeights();
    if(directed_){
        typedef graph_types<true>::Graph Graph;
        Graph graph;
        return init_with_graph<Node2VecSeqOp, Graph>(this, env, filename, graph);
    }
    else{
        typedef graph_types<false>::Graph Graph;
        Graph graph;
        return init_with_graph<Node2VecSeqOp, Graph>(this, env, filename, graph);
    }
}


void Node2VecSeqOp::PrecomputeWalk(int walk_idx, int start_node, random::SimplePhilox& gen){
    // First sample start node
    Alias a = node_alias_[start_node];
    int from_node;
    if(HasWeights()){
        from_node = sample_alias(a, gen);
    }
    else{
        auto i = gen.Uniform(a.idx.size());
        from_node = a.idx[i];
    }

    // Now sample using w2v distribution    
    int prev_node = start_node;
    auto w = precomputed_walks.matrix<int32>();
    w(walk_idx, 0) = start_node;
    w(walk_idx, 1) = from_node;
    //w[1] = from_node;
    for(int k=2; k < seq_size_; k++){
        assert(edge_alias_[from_node].find(prev_node) != edge_alias_[from_node].end());
        Alias* a = &(edge_alias_[from_node][prev_node]);
        int next_node = sample_alias(*a, gen);
        w(walk_idx, k) = (int32) next_node;
        prev_node = from_node; from_node = next_node;
    }
}

Status Node2VecSeqOp::Init(Env* env, const string& filename) {
  // std::cout << "Init" << std::endl;
    if (p_ == 0. || q_ == 0.) {
        return errors::InvalidArgument("The parameters p and q can't be 0.");
    }
    if (seq_size_ < 2) {
        return errors::InvalidArgument("The sequence size must be greater than two");
    }
    write_walk_idx = 0;
    cur_walk_idx = 0;

    precomputed_walks = Tensor(DT_INT32, TensorShape({PRECOMPUTE, seq_size_}));
    return MakeGraphTypeAndInit(env, filename);
}

std::vector<std::unordered_map<int, Alias>>* Node2VecSeqOp::getEdgeAlias(){
    return &edge_alias_;
}


Status RandWalkSeq::MakeGraphTypeAndInit(Env* env, const string& filename){
    bool has_weight = HasWeights();
    if(directed_){
      typedef graph_types<true>::Graph Graph;
      Graph graph;
      return init_with_graph<RandWalkSeq, Graph>(this, env, filename, graph);
    }
    else{
      typedef graph_types<false>::Graph Graph;
      Graph graph;
      return init_with_graph<RandWalkSeq, Graph>(this, env, filename, graph);
    }
}

void RandWalkSeq::PrecomputeWalk(int walk_idx, int start_node, random::SimplePhilox& gen){
  int node = start_node;
  auto w = precomputed_walks.matrix<int32>();
  w(walk_idx, 0) = start_node;
  //w[1] = from_node;
  for(int k=1; k < seq_size_; k++){
    Alias a = node_alias_[node];
    if(HasWeights()){
      node = sample_alias(a, gen);
    }
    else{
      auto i = gen.Uniform(a.idx.size());
      node = a.idx[i];
    }
    w(walk_idx, k) = (int32) node;
  }
}


Status RandWalkSeq::Init(Env* env, const string& filename) {
  if (seq_size_ < 2) {
    return errors::InvalidArgument("The sequence's size must be greater than two");
  }
  write_walk_idx = 0;
  cur_walk_idx = 0;
  precomputed_walks = Tensor(DT_INT32, TensorShape({PRECOMPUTE, seq_size_}));
  return MakeGraphTypeAndInit(env, filename);
}

// template<> Status init_with_graph<RandWalkSeq, typename graph_types<true>::Graph>;
// template<> Status init_with_graph<RandWalkSeq, typename graph_types<false>::Graph>;


REGISTER_KERNEL_BUILDER(Name("RandWalkSeq").Device(DEVICE_CPU), RandWalkSeq);

REGISTER_KERNEL_BUILDER(Name("Node2VecSeq").Device(DEVICE_CPU), Node2VecSeqOp);

} // Namespace