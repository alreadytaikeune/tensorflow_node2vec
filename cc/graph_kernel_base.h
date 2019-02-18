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
#ifndef GRAPH_KERNEL_BASE_H
#define GRAPH_KERNEL_BASE_H

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

#include <boost/graph/adjacency_list.hpp>

#include "sampling.h"


using namespace tensorflow;
using namespace std;

const int PRECOMPUTE = 30000;
const int LOW_WATER_MARK = 100;


namespace gseq{

struct VertexProperty
{
    std::string id;
};


struct EdgeProperty{
  double weight;
};

template<bool D>
struct directed_type{
  typedef boost::undirectedS type;
};

template<>
struct directed_type<true>{
  typedef boost::directedS type;
};


template<bool D>
struct graph_types{
  typedef boost::adjacency_list<boost::vecS, boost::vecS, typename directed_type<D>::type, VertexProperty, EdgeProperty> Graph;
};



template<typename G> int setup_node_alias(G graph, std::vector<Alias>& node_alias, std::vector<int32>& valid_nodes, bool has_weights) {
    int32 nb_vertices = static_cast<int32>(boost::num_vertices(graph));
    node_alias.resize(nb_vertices);
    int alias_idx = 0;
    for(int i=0; i<nb_vertices; ++i){
        typename G::adjacency_iterator vit, vend;
        std::tie(vit, vend) = boost::adjacent_vertices(i, graph);
        int nb_neighbors = std::distance(vit, vend);
        if(nb_neighbors > 0){
            valid_nodes.push_back(i);
            // cout << "pushed back " << i << " to valid nodes" << endl;
            double sum_weights=0;
            for(auto it = vit; it != vend; ++it){
                if(has_weights){
                    auto e = boost::edge(i,*it, graph).first;
                    double weight = graph[e].weight;
                    sum_weights += weight;
                    node_alias[i].probas.push_back(weight);
                }
                node_alias[i].idx.push_back(*it);
                alias_idx++;
            }
            if(has_weights){
                setup_alias_vectors(node_alias[i], sum_weights);
            }
        }
    }
}


class BaseGraphKernel : public OpKernel {
public:
    explicit BaseGraphKernel(OpKernelConstruction* ctx);

    void Compute(OpKernelContext* ctx) override;

    bool HasWeights();
    void SetHasWeights(bool b);

    std::vector<Alias>* getNodeAlias();
    std::vector<int32>* getValidNodes();
    const std::string& getWeightAttrName();
    Tensor& getNodeId();

    void InitNodeId(int nb);

    void NextWalk(OpKernelContext* ctx, Tensor& walk, int i) EXCLUSIVE_LOCKS_REQUIRED(mu_);

    void PrecomputeWalks(int write_idx, int start_idx, int end_idx);
    virtual Status MakeGraphTypeAndInit(Env* env, const string& filename) = 0;

    virtual Status Init(Env* env, const string& filename) = 0;
    virtual void PrecomputeWalk(int walk_idx, int start_node, random::SimplePhilox& gen) = 0;
protected:
    int32 batchsize_ = 128;
    int32 seq_size_ = 0;
    int32 graph_size_ = 0;
    bool directed_ = false;
    std::string weight_attr_name_;

    Tensor node_id_;
    std::vector<int32> valid_nodes_;

    tensorflow::mutex mu_;
    GuardedPhiloxRandom guarded_philox_ GUARDED_BY(mu_);
    int32 current_epoch_ GUARDED_BY(mu_) = -1;
    int32 total_seq_generated_ GUARDED_BY(mu_) = 0;
    int32 current_node_idx_ GUARDED_BY(mu_) = 0;
    Tensor precomputed_walks;
    int cur_walk_idx;
    int write_walk_idx;
    int num_threads_;
    int nb_valid_nodes_ = 0;
    bool has_weights_ = false;
    std::vector<Alias> node_alias_;

};


} // Namespace

#endif // GRAPH_KERNEL_BASE_H