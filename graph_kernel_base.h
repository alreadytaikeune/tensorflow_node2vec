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


class BaseGraphKernel : public OpKernel {
 public:
  explicit BaseGraphKernel(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 protected:

  int32 seq_size_ = 0;
  int32 graph_size_ = 0;
  bool directed_ = false;

  Tensor node_id_;
  std::vector<int32> valid_nodes_;

  mutex mu_;
  GuardedPhiloxRandom guarded_philox_ GUARDED_BY(mu_);
  int32 current_epoch_ GUARDED_BY(mu_) = -1;
  int32 total_seq_generated_ GUARDED_BY(mu_) = 0;
  int32 current_node_idx_ GUARDED_BY(mu_) = 0;
  Tensor precomputed_walks;
  int cur_walk_idx;
  int write_walk_idx;
  int num_threads_;
  int nb_valid_nodes_ = 0;


  void NextWalk(OpKernelContext* ctx, Tensor& walk) EXCLUSIVE_LOCKS_REQUIRED(mu_);


  void PrecomputeWalks(int write_idx, int start_idx, int end_idx);

  virtual Status Init(Env* env, const string& filename) = 0;
  virtual void PrecomputeWalk(int walk_idx, int start_node, random::SimplePhilox& gen) = 0;
};
