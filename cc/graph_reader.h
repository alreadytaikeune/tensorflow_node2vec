#ifndef GRAPH_READER_H
#define GRAPH_READER_H

#include <cassert>
#include <map>

#include <tensorflow/core/lib/core/status.h>
#include <boost/filesystem.hpp>
#include <boost/graph/graphml.hpp>
#include "graph_kernel_base.h"

using namespace tensorflow;

namespace gseq{

template<typename Graph>
void read_graphml(std::istream& data_stream, Graph& graph, boost::dynamic_properties& dp){
    boost::read_graphml(data_stream, graph, dp);
}


class EdgeListReader {
public:
    EdgeListReader(boost::mutate_graph& g) 
        : m_g(g) { }

    void ReadEdgeList(std::istream& data_stream, bool has_weights, const std::string& weight_attr_name){
        std::string line;
        std::getline(data_stream, line);
        while(line[0] == '#' && std::getline(data_stream, line)){
        }
        assert(data_stream && "unexpected EOF while reading edge list");
        bool saw_space=false;
        int nb_spaces = 0;
        for(char& x : line){
            if(x == ' ')
                saw_space = true;
            else{
                nb_spaces += saw_space;
                saw_space = false;
            }
        }
        assert(((has_weights && nb_spaces == 2) || !has_weights) && "Has weights: True. Expected line format 'node1 node2 weight'");
        assert(((!has_weights && nb_spaces == 1) || has_weights) && "Has weights: False. Expected line format 'node1 node2'");

        std::istringstream line_stream;
        do{
            if(line[0] != '#'){
                line_stream.str(line);
                try{
                    HandleLine(line_stream, has_weights, weight_attr_name);
                } catch(std::istream::failure &E){
                    cerr << "Line " << line << " has unexpected format" << endl;
                    throw E;
                }
                line_stream.clear();
            } 
        } while(std::getline(data_stream, line));
    }

    void HandleLine(std::istringstream& line, bool has_weights, const std::string& weight_attr_name){
        std::string node1, node2, weight;
        line >> node1 >> node2;
        if(has_weights)
            line >> weight;
        HandleEdge(node1, node2, has_weights, weight, weight_attr_name);
    }

    void HandleVertex(const std::string& v){
        bool is_new = m_vertex.find(v) == m_vertex.end();
        if (is_new){
            auto node_idx = m_g.do_add_vertex();
            m_vertex[v] = node_idx;
            m_g.set_vertex_property("id", node_idx, v, "string");
        }
    }


    void HandleEdge(const std::string& u, const std::string& v, bool has_weights, const std::string& weight, const std::string& weight_attr_name){
        HandleVertex(u); HandleVertex(v);
        boost::any source, target;
        source = m_vertex[u];
        target = m_vertex[v];

        boost::any edge;
        bool added;
        boost::tie(edge, added) = m_g.do_add_edge(source, target);
        if(!added)
            BOOST_THROW_EXCEPTION(boost::bad_parallel_edge(u, v));
        if(has_weights)
            m_g.set_edge_property(weight_attr_name, edge, weight, "float");
        m_edge_idx++;
    }

private:
    boost::mutate_graph& m_g;
    std::map<std::string, boost::any> m_vertex;
    size_t m_edge_idx = 0;
};


template<typename Graph>
void read_edgelist(std::istream& data_stream, Graph& graph, boost::dynamic_properties& dp, bool has_weights, const std::string& weight_attr_name){
    boost::mutate_graph_impl<Graph> mg(graph, dp);
    EdgeListReader reader(mg);
    reader.ReadEdgeList(data_stream, has_weights, weight_attr_name);
}


template<typename Graph>
void read_edgelist(std::istream& data_stream, Graph& graph, boost::dynamic_properties& dp){
    const string s;
    read_edgelist(data_stream, graph, dp, false, s);
}



template <typename Graph>
Status read_graph(Env* env, const std::string& filename, Graph& graph, boost::dynamic_properties& dp, bool has_weight, const std::string& weight_attr_name){
    string data;
    TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &data));
    std::istringstream data_stream;
    data_stream.str(data);
    boost::filesystem::path path = filename;
    std::string ext = path.extension().string();
    if(ext == ".graphml"){
        gseq::read_graphml(data_stream, graph, dp);
    }
    else{
        read_edgelist(data_stream, graph, dp, has_weight, weight_attr_name);
    }
    return Status::OK();
}


} // Namespace

#endif // GRAPH_READER_H