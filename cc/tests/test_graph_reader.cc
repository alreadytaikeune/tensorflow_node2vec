#include <iostream>
#include <fstream>
#include <cassert>

#include "graph_reader.h"
#include "graph_kernel_base.h"

using namespace gseq;

typedef typename graph_types<true>::Graph DGraph;
typedef typename graph_types<false>::Graph Graph;

void test_read_graphml(){
    std::ifstream fin("../../data/miserables.graphml");
    Graph g;
    boost::dynamic_properties dp(boost::ignore_other_properties);
    dp.property("id", boost::get(&VertexProperty::id, g));
    gseq::read_graphml(fin, g, dp);
    int32 nb_vertices = static_cast<int32>(boost::num_vertices(g));
    int32 nb_edges = static_cast<int32>(boost::num_edges(g));
    assert(nb_vertices == 77 && nb_edges == 254);
    cout << "test read graphml ok" << endl;
}


void test_read_edgelist(){
    std::ifstream fin("../../data/miserables_edgelist");
    Graph g;
    boost::dynamic_properties dp(boost::ignore_other_properties);
    dp.property("id", boost::get(&VertexProperty::id, g));
    read_edgelist(fin, g, dp);
    int32 nb_vertices = static_cast<int32>(boost::num_vertices(g));
    int32 nb_edges = static_cast<int32>(boost::num_edges(g));
    assert(nb_vertices == 77 && nb_edges == 254);
    cout << "test read edgelist ok" << endl;
}


int main(){
    test_read_graphml();
    test_read_edgelist();
    return 0;
}