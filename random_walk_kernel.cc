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
#include <algorithm>

#include <ctime>
#include <iostream>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/util/work_sharder.h"


#include <boost/graph/graphml.hpp>
#include <boost/graph/adjacency_list.hpp>

#include "sampling.h"


using namespace tensorflow;


const int PRECOMPUTE = 30000;
const int LOW_WATER_MARK = 100;


struct VertexProperty
{
    std::string id;
};


struct EdgeProperty{
  double weight;
};

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, VertexProperty, EdgeProperty> Graph;


double time_ms(std::clock_t start, std::clock_t end){
  return (end - start) / (double)(CLOCKS_PER_SEC / 1000);
}


class RandWalkSeq : public OpKernel {
 public:
  explicit RandWalkSeq(OpKernelConstruction* ctx)
      : OpKernel(ctx){
    string filename;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("size", &seq_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("directed", &directed_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("weights_attribute", &weight_attr_name_));
    OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    num_threads_ = worker_threads.num_threads;
    guarded_philox_.Init(0, 0);
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor epoch(DT_INT32, TensorShape({}));
    Tensor total(DT_INT32, TensorShape({}));
    Tensor nb_valid_nodes(DT_INT32, TensorShape({}));
    Tensor walk(DT_INT32, TensorShape({seq_size_}));
    {
      mutex_lock l(mu_);
      NextWalk(ctx, walk);
      epoch.scalar<int32>()() = current_epoch_;
      total.scalar<int32>()() = total_seq_generated_;
    }
    nb_valid_nodes.scalar<int32>()() = nb_valid_nodes_;
    ctx->set_output(0, node_id_);
    ctx->set_output(1, walk);
    ctx->set_output(2, epoch);
    ctx->set_output(3, total);
    ctx->set_output(4, nb_valid_nodes);

  }


 private:

  int32 seq_size_ = 0;

  int32 graph_size_ = 0;
  std::string weight_attr_name_;

  bool directed_ = false;
  Tensor node_id_;

  std::vector<Alias> node_alias_;
  std::vector<int32> valid_nodes_;

  mutex mu_;
  mutex shard_mutex_;
  GuardedPhiloxRandom guarded_philox_ GUARDED_BY(mu_);
  int32 current_epoch_ GUARDED_BY(mu_) = -1;
  int32 total_seq_generated_ GUARDED_BY(mu_) = 0;
  int32 current_node_idx_ GUARDED_BY(mu_) = 0;
  Tensor precomputed_walks;
  int cur_walk_idx;
  int write_walk_idx;
  int num_threads_;
  int nb_valid_nodes_ = 0;


  void NextWalk(OpKernelContext* ctx, Tensor& walk) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    int available = (write_walk_idx + PRECOMPUTE - cur_walk_idx) % PRECOMPUTE;
    if(available <= LOW_WATER_MARK){
      int start = write_walk_idx;
      int end = cur_walk_idx-1;
      if(end <= start)
        end += PRECOMPUTE;
      auto fn = [this](int64 s, int64 e){
        PrecomputeWalks(write_walk_idx+s, write_walk_idx+e);
      };

      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      // worker_threads.num_threads

      Shard(1, worker_threads.workers,
            end-start, 20000000,
            fn);
      write_walk_idx+=(end-start);
      write_walk_idx%=PRECOMPUTE;
    }

    auto w = walk.flat<int32>();
    auto pre = precomputed_walks.matrix<int32>().chip<0>(cur_walk_idx);
    w=pre;

    cur_walk_idx++;
    total_seq_generated_++;
    cur_walk_idx%=PRECOMPUTE;
    current_epoch_ = total_seq_generated_/valid_nodes_.size();
  }


  void PrecomputeWalks(int start_idx, int end_idx){
    random::PhiloxRandom phi = guarded_philox_.ReserveSamples128(seq_size_*2);
    std::vector<int32> starts;
    int N = valid_nodes_.size(); // Precompute once and for all?
    for(int i=start_idx; i<end_idx; i++){
      starts.push_back(valid_nodes_[current_node_idx_]);
      current_node_idx_++;
      current_node_idx_%=N;
    }
    
    random::SimplePhilox gen(&phi);
    for(int i=start_idx; i<end_idx; i++){
      PrecomputeWalk(i%PRECOMPUTE, starts[i-start_idx], gen);
    }
  }

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
      return errors::InvalidArgument("The sequence size must be greater than two");
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
          auto e = boost::edge(i,*it, graph).first;
          double weight = graph[e].weight;
          sum_weights += weight;
          node_alias_[i].probas.push_back(weight);
          node_alias_[i].idx.push_back(*it);
          alias_idx++;
        }
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
