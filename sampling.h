#include <vector>
#include <queue>

#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/util/guarded_philox_random.h"

using namespace tensorflow;


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

int sample_alias(Alias& alias, random::SimplePhilox& gen){
    int N = alias.probas.size();
    int v = gen.Uniform(N);
    double x = gen.RandDouble();
    if(x < alias.probas[v]){
        return alias.idx[v];
    }
    return alias.idx[alias.aliases[v]];
}
