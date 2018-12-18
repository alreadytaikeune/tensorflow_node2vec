#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <iterator>
#include <vector>
#include <cassert>
#include <queue>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/guarded_philox_random.h"

#include <boost/graph/graphml.hpp>
#include <boost/graph/adjacency_list.hpp>

using namespace tensorflow;


struct VertexProperty
{
    std::string id ;
};


typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, VertexProperty, boost::no_property> Graph;

typedef struct Alias {
  std::vector<double> probas;
  std::vector<int> aliases;
  std::vector<int> idx;
} Alias;


void setup_alias_vectors(Alias& alias, double norm){
    int N = alias.probas.size();
    assert(alias.probas.size() == alias.idx.size());
    alias.aliases.resize(N);
    std::queue<int> big;
    std::queue<int> small;
    double f = N/norm;
    for(int i=0; i<N; i++){
        alias.probas[i] = alias.probas[i]*f;
        if(alias.probas[i] < 1.)
          small.push(i);
        else
          big.push(i);
    }
    while(!(big.empty() || small.empty())){
        int s = small.front(); small.pop();
        int b = big.front(); big.pop();
        alias.aliases[s] = b;
        double ptot = alias.probas[s]*N + alias.probas[b]*N - 1.;
        alias.probas[b] = ptot;
        if(ptot < 1.){
          small.push(b);
        }
        else{
          big.push(b);
        }
    }
}


int sample_alias(Alias& alias, random::SimplePhilox& rng_){
    int N = alias.probas.size();
    int v = rng_.Uniform(N);
    double x = rng_.RandDouble();
    if(x < alias.probas[v]){
        return alias.idx[v];
    }
    return alias.idx[alias.aliases[v]];
}


class Node2VecSeqOp : public OpKernel {
 public:
  explicit Node2VecSeqOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), rng_(&philox_) {
    string filename;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("size", &seq_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("p", &p_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("q", &q_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("directed", &directed_));
    OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor epoch(DT_INT32, TensorShape({}));
    Tensor total(DT_INT32, TensorShape({}));
    Tensor walk(DT_INT32, TensorShape({seq_size_}));
    {
      mutex_lock l(mu_);
      NextWalk(walk);
      epoch.scalar<int32>()() = current_epoch_;
      total.scalar<int32>()() = total_seq_generated_;
    }
    ctx->set_output(0, node_id_);
    ctx->set_output(1, walk);
    ctx->set_output(2, epoch);
    ctx->set_output(3, total);

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

  mutex mu_;
  random::PhiloxRandom philox_ GUARDED_BY(mu_);
  random::SimplePhilox rng_ GUARDED_BY(mu_);
  int32 current_epoch_ GUARDED_BY(mu_) = -1;
  int32 total_seq_generated_ GUARDED_BY(mu_) = 0;
  int32 current_node_idx_ GUARDED_BY(mu_) = 0;


  void NextWalk(Tensor& walk) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if(current_node_idx_ == 0)
      current_epoch_++;
    int start = current_node_idx_;
    int counter=0;
    while(node_alias_[start].size()==0 && counter < graph_size_){
      start++; start%=graph_size_; counter++;
      if(start==0)
        current_epoch_++;
    }
    current_node_idx_ = (start+1)%graph_size_;
    total_seq_generated_++;
    if(node_alias_[start].size()==0){
      throw new std::exception();
    }
    
    // sample first neighbor
    int i=rng_.Uniform(node_alias_[start].size());
    int from_node = node_alias_[start][i];
    int prev_node = start;
    auto w = walk.flat<int32>();
    w(0) = start; w(1) = from_node;
    for(int k=2; k < seq_size_; k++){
        Alias* a = &edge_alias_[from_node][prev_node];
        int next_node = sample_alias(*a, rng_);
        w(k) = (int32) next_node;
        prev_node = from_node; from_node = next_node;
    }
  }

  Status Init(Env* env, const string& filename) {
    std::cout << "Init" << std::endl;
    if (p_ == 0. || q_ == 0.) {
      return errors::InvalidArgument("The parameters p and q can't be 0.");
    }
    if (seq_size_ < 2) {
      return errors::InvalidArgument("The sequence size must be greater than two");
    }


    Graph graph;
    boost::dynamic_properties dp(boost::ignore_other_properties);
    dp.property("id", boost::get(&VertexProperty::id, graph));

    std::cout << "Reading the graph" << std::endl;
    {
      string data;
      TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &data));
      std::istringstream data_stream;
      data_stream.str(data);
      boost::read_graphml(data_stream, graph, dp);
    }
    int32 nb_vertices = static_cast<int32>(boost::num_vertices(graph));
    int32 nb_edges = static_cast<int32>(boost::num_edges(graph));

    std::cout << "nb vertices: " << nb_vertices << " nb edges " << nb_edges << std::endl;

    node_id_ = Tensor(DT_STRING, TensorShape({nb_vertices}));
    for(int i=0; i<nb_vertices; ++i){
      node_id_.flat<string>()(i) = graph[i].id;
    }


    // ----- Computing node aliases -----
    std::cout << "Setting node aliases" << std::endl;
    {
      int alias_idx = 0;
      node_alias_.resize(nb_vertices);
      for(int i=0; i<nb_vertices; ++i){
        Graph::adjacency_iterator vit, vend;
        std::tie(vit, vend) = boost::adjacent_vertices(i, graph);
        int nb_neighbors = std::distance(vit, vend);
        for(auto it = vit; it != vend; ++it){
          node_alias_[i].push_back(*it);
          alias_idx++;
        }
      }
    }
    // ----- Done -----

    // ----- Computing edges aliases -----
    std::cout << "Setting edge aliases" << std::endl;
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
