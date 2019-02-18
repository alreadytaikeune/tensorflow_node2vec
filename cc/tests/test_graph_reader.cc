#include <iostream>
#include <fstream>
#include <cassert>

#include "graph_reader.h"
#include "graph_kernel_base.h"

using namespace gseq;

typedef typename graph_types<true>::Graph DGraph;
typedef typename graph_types<false>::Graph Graph;


template<typename G>
void read_graph(std::string&& fname, G& g, bool graphml){
    std::ifstream fin(fname);
    boost::dynamic_properties dp(boost::ignore_other_properties);
    dp.property("id", boost::get(&VertexProperty::id, g));
    if(graphml)
        gseq::read_graphml(fin, g, dp);
    else
        gseq::read_edgelist(fin, g, dp);
}



template<typename G>
void test_nb_vertices_edges(G& g, int nbv, int nbe){
    int32 nb_vertices = static_cast<int32>(boost::num_vertices(g));
    int32 nb_edges = static_cast<int32>(boost::num_edges(g));
    cout << nb_vertices << " " << nb_edges << endl;
    assert(nb_vertices == nbv && nb_edges == nbe);
}

void test_read_graphml(){
    Graph g;
    read_graph("../../data/miserables.graphml", g, true);
    test_nb_vertices_edges(g, 77, 254);
    cout << "test read graphml ok" << endl;
}


void test_read_edgelist(){
    Graph g;
    read_graph("../../data/miserables_edgelist", g, false);
    test_nb_vertices_edges(g, 77, 254);
    cout << "test read edgelist ok" << endl;
}


void test_read_edgelist_directed(){
    DGraph g;
    read_graph("../../data/miserables_edgelist", g, false);
    test_nb_vertices_edges(g, 77, 254);
    cout << "test read edgelist ok" << endl;
}

int main(){
    test_read_graphml();
    test_read_edgelist();
    test_read_edgelist_directed();
    return 0;
}