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
#include <iterator>
#include <vector>
#include <cassert>
#include <algorithm>

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

using namespace tensorflow;


class RandWalkSeq : public BaseGraphKernel {
 public:
  explicit RandWalkSeq(OpKernelConstruction* ctx)
      : BaseGraphKernel(ctx){
    string filename;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("size", &seq_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("directed", &directed_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("weights_attribute", &weight_attr_name_));
    OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));
  }


 protected:
  std::string weight_attr_name_;
  std::vector<Alias> node_alias_;

  bool HasWeights(){
    return weight_attr_name_.size() > 0;
  }

  void PrecomputeWalk(int walk_idx, int start_node, random::SimplePhilox& gen){
    int node = start_node;
    auto w = precomputed_walks.matrix<int32>();
    w(walk_idx, 0) = start_node;
    //w[1] = from_node;
    for(int k=1; k < seq_size_; k++){
      Alias* a = &node_alias_[node];
      if(HasWeights()){
        node = sample_alias(*a, gen);
      }
      else{
        auto i = gen.Uniform(a->idx.size());
        node = a->idx[i];
      }
      w(walk_idx, k) = (int32) node;
    }
  }


  Status Init(Env* env, const string& filename) {

    if (seq_size_ < 2) {
      return errors::InvalidArgument("The sequence's size must be greater than two");
    }
    write_walk_idx = 0;
    cur_walk_idx = 0;

    precomputed_walks = Tensor(DT_INT32, TensorShape({PRECOMPUTE, seq_size_}));

    Graph graph;
    boost::dynamic_properties dp(boost::ignore_other_properties);
    dp.property("id", boost::get(&VertexProperty::id, graph));
    if(weight_attr_name_.size() > 0){
      dp.property(weight_attr_name_, boost::get(&EdgeProperty::weight, graph));
    }

    {
      string data;
      TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &data));
      std::istringstream data_stream;
      data_stream.str(data);
      boost::read_graphml(data_stream, graph, dp);
    }
    int32 nb_vertices = static_cast<int32>(boost::num_vertices(graph));
    int32 nb_edges = static_cast<int32>(boost::num_edges(graph));

    node_id_ = Tensor(DT_STRING, TensorShape({nb_vertices}));
    for(int i=0; i<nb_vertices; ++i){
      node_id_.flat<string>()(i) = graph[i].id;
    }


    // ----- Computing node aliases -----
    // std::cout << "Setting node aliases" << std::endl;
    node_alias_.resize(nb_vertices);
    int alias_idx = 0;
    for(int i=0; i<nb_vertices; ++i){
      Graph::adjacency_iterator vit, vend;
      std::tie(vit, vend) = boost::adjacent_vertices(i, graph);
      int nb_neighbors = std::distance(vit, vend);
      if(nb_neighbors > 0){
        valid_nodes_.push_back(i);
        double sum_weights=0;
        for(auto it = vit; it != vend; ++it){
          if(HasWeights()){
            auto e = boost::edge(i,*it, graph).first;
            double weight = graph[e].weight;
            sum_weights += weight;
            node_alias_[i].probas.push_back(weight);
          }
          node_alias_[i].idx.push_back(*it);
          alias_idx++;
        }
        if(HasWeights())
          setup_alias_vectors(node_alias_[i], sum_weights);
      }
    }

    nb_valid_nodes_ = valid_nodes_.size();
    // ----- Done -----
    std::cout << "Has weights: " << HasWeights() << std::endl;
    graph.clear();

    graph_size_ = nb_vertices;
    return Status::OK();
  }
};

REGISTER_KERNEL_BUILDER(Name("RandWalkSeq").Device(DEVICE_CPU), RandWalkSeq);
