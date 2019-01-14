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
#include <queue>

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


typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, VertexProperty, boost::no_property> Graph;



double time_ms(std::clock_t start, std::clock_t end){
  return (end - start) / (double)(CLOCKS_PER_SEC / 1000);
}

class Node2VecSeqOp : public OpKernel {
 public:
  explicit Node2VecSeqOp(OpKernelConstruction* ctx)
      : OpKernel(ctx){
    string filename;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("size", &seq_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("p", &p_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("q", &q_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("directed", &directed_));
    OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    num_threads_ = worker_threads.num_threads;
    std::cout << "num threads is " << num_threads_ << std::endl;
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
  float p_ = 1.;
  float q_ = 1.;
  int32 graph_size_ = 0;

  bool directed_ = false;
  Tensor node_id_;

  std::vector<std::vector<int>> node_alias_;
  std::vector<std::unordered_map<int, Alias>> edge_alias_;
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
      // std::cout << "Sharding work start " << start << " end " << end << std::endl;

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
    random::PhiloxRandom phi;
    std::vector<int32> starts;
    int N = valid_nodes_.size(); // Precompute once and for all?
    {
      mutex_lock l(shard_mutex_);
      phi = guarded_philox_.ReserveSamples128((end_idx-start_idx) + seq_size_*2);
      for(int i=start_idx; i<end_idx; i++){
        starts.push_back(valid_nodes_[current_node_idx_]);
        current_node_idx_++;
        current_node_idx_%=N;
      }
    }
    
    random::SimplePhilox gen(&phi);
    for(int i=start_idx; i<end_idx; i++){
      PrecomputeWalk(i%PRECOMPUTE, starts[i-start_idx], gen);
    }
  }

  void PrecomputeWalk(int walk_idx, int start_node, random::SimplePhilox& gen){
    int i=gen.Uniform(node_alias_[start_node].size());
    int from_node = node_alias_[start_node][i];
    int prev_node = start_node;
    auto w = precomputed_walks.matrix<int32>();
    w(walk_idx, 0) = start_node;
    w(walk_idx, 1) = from_node;
    //w[1] = from_node;
    for(int k=2; k < seq_size_; k++){
        Alias* a = &edge_alias_[from_node][prev_node];
        int next_node = sample_alias(*a, gen);
        w(walk_idx, k) = (int32) next_node;
        prev_node = from_node; from_node = next_node;
    }
  }

  Status Init(Env* env, const string& filename) {
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

    Graph graph;
    boost::dynamic_properties dp(boost::ignore_other_properties);
    dp.property("id", boost::get(&VertexProperty::id, graph));

    // std::cout << "Reading the graph" << std::endl;
    {
      string data;
      TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &data));
      std::istringstream data_stream;
      data_stream.str(data);
      boost::read_graphml(data_stream, graph, dp);
    }
    int32 nb_vertices = static_cast<int32>(boost::num_vertices(graph));
    int32 nb_edges = static_cast<int32>(boost::num_edges(graph));

    // std::cout << "nb vertices: " << nb_vertices << " nb edges " << nb_edges << std::endl;

    node_id_ = Tensor(DT_STRING, TensorShape({nb_vertices}));
    for(int i=0; i<nb_vertices; ++i){
      node_id_.flat<string>()(i) = graph[i].id;
    }


    // ----- Computing node aliases -----
    // std::cout << "Setting node aliases" << std::endl;
    {
      int alias_idx = 0;
      node_alias_.resize(nb_vertices);
      for(int i=0; i<nb_vertices; ++i){
        Graph::adjacency_iterator vit, vend;
        std::tie(vit, vend) = boost::adjacent_vertices(i, graph);
        int nb_neighbors = std::distance(vit, vend);
        if(nb_neighbors > 0)
          valid_nodes_.push_back(i);
        for(auto it = vit; it != vend; ++it){
          node_alias_[i].push_back(*it);
          alias_idx++;
        }
      }
    }

    nb_valid_nodes_ = valid_nodes_.size();
    // ----- Done -----

    // ----- Computing edges aliases -----
    // std::cout << "Setting edge aliases" << std::endl;
    int total_entries = 0;
    for(int target=0; target<nb_vertices; ++target){
        if(target % 1000 == 0)
          std::cout << target << "/" << nb_vertices << std::endl;
        Graph::adjacency_iterator vit, vend;
        Graph::adjacency_iterator sit, send;
        std::tie(vit, vend) = boost::adjacent_vertices(target, graph);
        std::unordered_map<int, Alias> nmap;

        for(auto source_it = vit; source_it != vend; ++source_it){
          std::unordered_set<int> source_neighbors;
          int source = *source_it;
          std::tie(sit, send) = boost::adjacent_vertices(source, graph);
          source_neighbors.insert(sit, send);
          double sum_weights=0;
          for(auto it = vit; it != vend; ++it){
            double weight = 1.;
            if(*it == source)
              weight = 1./p_;
            auto prev_neigh = source_neighbors.find(*it);
            if(prev_neigh == source_neighbors.end())
                weight = 1./q_;
            sum_weights += weight;
            nmap[source].probas.push_back(weight);
            nmap[source].idx.push_back(*it);

            total_entries += 1;
          }
          setup_alias_vectors(nmap[source], sum_weights);
        }
        edge_alias_.push_back(nmap);
    }

    std::cout << total_entries << " in the edge alias" << std::endl;
    graph.clear();

    graph_size_ = nb_vertices;
    return Status::OK();
  }
};

REGISTER_KERNEL_BUILDER(Name("Node2VecSeq").Device(DEVICE_CPU), Node2VecSeqOp);
