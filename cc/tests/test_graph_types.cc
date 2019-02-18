#include <iostream>
#include <cassert>
#include "graph_reader.h"
#include "graph_kernel_base.h"

using namespace gseq;

typedef typename graph_types<true>::Graph DGraph;
typedef typename graph_types<false>::Graph Graph;

template <typename G>
bool is_directed(G &g)
{
  // Typedef of an object whose constructor returns "directedness" of the graph object.
  typedef typename boost::graph_traits<G>::directed_category Cat;
  // The function boost::detail::is_directed() returns "true" if the graph object is directed.
  return boost::detail::is_directed(Cat());
}


void test_graph_types(){
    DGraph dg;
    Graph g;
    assert(is_directed(dg) && !is_directed(g));
}

int main(){
    test_graph_types();
    cout << "test graph types OK" << endl;
    return 0;
}