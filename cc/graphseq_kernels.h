#ifndef GRAPHSEQ_KERNELS_H
#define GRAPHSEQ_KERNELS_H


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/util/guarded_philox_random.h"


#include "sampling.h"
#include "graph_kernel_base.h"
#include "graph_reader.h"

namespace gseq{


class Node2VecSeqOp : public BaseGraphKernel {
public:
    explicit Node2VecSeqOp(OpKernelConstruction* ctx) : BaseGraphKernel(ctx){
        OP_REQUIRES_OK(ctx, ctx->GetAttr("p", &p_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("q", &q_));
        string filename;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename));
        OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));
    }

    std::vector<std::unordered_map<int, Alias>>* getEdgeAlias();

    float p_ = 1.;
    float q_ = 1.;
private:
    std::vector<std::unordered_map<int, Alias>> edge_alias_;
protected:
    virtual Status Init(Env* env, const string& filename);
    virtual Status MakeGraphTypeAndInit(Env* env, const string& filename);
    virtual void PrecomputeWalk(int walk_idx, int start_node, random::SimplePhilox& gen);
};



class RandWalkSeq : public BaseGraphKernel {
public:
    explicit RandWalkSeq(OpKernelConstruction* ctx)
      : BaseGraphKernel(ctx){
        string filename;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename));
        OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));
    }

protected:
    virtual Status Init(Env* env, const string& filename);
    virtual Status MakeGraphTypeAndInit(Env* env, const string& filename);
    virtual void PrecomputeWalk(int walk_idx, int start_node, random::SimplePhilox& gen);

};


template<typename T, typename G> struct AliasStructureSetter{
    void Setup(T* kernel, G& graph);
};


template<typename G>
struct AliasStructureSetter<Node2VecSeqOp, G>{

    void Setup(Node2VecSeqOp* kernel, G &graph){
        auto edge_alias = kernel->getEdgeAlias();
        auto node_alias = kernel->getNodeAlias();
        auto valid_nodes = kernel->getValidNodes();
        setup_node_alias(graph, *node_alias, *valid_nodes, kernel->HasWeights());
        int total_entries = 0;
        int32 nb_vertices = static_cast<int32>(boost::num_vertices(graph));
        edge_alias->resize(nb_vertices);
        for(int target=0; target<nb_vertices; ++target){
            if(target % 1000 == 0)
                std::cout << target << "/" << nb_vertices << std::endl;
            typename G::adjacency_iterator vit, vend;
            typename G::adjacency_iterator sit, send;
            std::tie(vit, vend) = boost::adjacent_vertices(target, graph);
            std::unordered_map<int, Alias>& nmap = (*edge_alias)[target];

            for(auto source_it = vit; source_it != vend; ++source_it){
                std::unordered_set<int> source_neighbors;
                int source = *source_it;
                std::tie(sit, send) = boost::adjacent_vertices(source, graph);
                source_neighbors.insert(sit, send);
                double sum_weights=0;
                for(auto it = vit; it != vend; ++it){
                    double weight = 1.;
                    if(kernel->HasWeights()){
                        auto e = boost::edge(*vit,*it, graph).first;
                        weight = graph[e].weight;
                    }
                    if(*it == source)
                        weight *= 1./kernel->p_;
                    else{
                        auto prev_neigh = source_neighbors.find(*it);
                        if(prev_neigh == source_neighbors.end())
                            weight *= 1./kernel->q_;
                    }
                    sum_weights += weight;
                    nmap[source].probas.push_back(weight);
                    nmap[source].idx.push_back(*it);

                    total_entries += 1;
                }
                setup_alias_vectors(nmap[source], sum_weights);
            }
        }
    }  
};


template<typename G>
struct AliasStructureSetter<RandWalkSeq, G>{
    void Setup(RandWalkSeq* kernel, G &graph){
        auto node_alias = kernel->getNodeAlias();
        auto valid_nodes = kernel->getValidNodes();
        setup_node_alias(graph, *node_alias, *valid_nodes, kernel->HasWeights());
    }
  
};


template<typename T, typename G> Status init_with_graph(T* kernel, Env* env, const string& filename, G& graph){
    boost::dynamic_properties dp(boost::ignore_other_properties);
    dp.property("id", boost::get(&VertexProperty::id, graph));
    if(kernel->HasWeights()){
        dp.property(kernel->getWeightAttrName(), boost::get(&EdgeProperty::weight, graph));
    }
    // std::cout << "Reading the graph" << std::endl;
    read_graph(env, filename, graph, dp, kernel->HasWeights(), kernel->getWeightAttrName());
    cout << "successfully read graph" << endl;
    int32 nb_vertices = static_cast<int32>(boost::num_vertices(graph));
    int32 nb_edges = static_cast<int32>(boost::num_edges(graph));
    kernel->InitNodeId(nb_vertices);
    std::cout << "nb vertices: " << nb_vertices << " nb edges " << nb_edges << std::endl;

    Tensor& node_id = kernel->getNodeId();
    for(int i=0; i<nb_vertices; ++i){
        node_id.flat<string>()(i) = graph[i].id;
    }

    AliasStructureSetter<T, G> setter;
    setter.Setup(kernel, graph);
    // graph_size_ = nb_vertices;
    graph.clear();
    return Status::OK();
}


} // Namespace

#endif // GRAPHSEQ_KERNELS_H