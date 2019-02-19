#include <iostream>
#include "sampling.h"

using namespace std;

namespace gseq{


void setup_alias_vectors(Alias& alias, float norm){
    int N = alias.probas.size();
    assert(alias.probas.size() == alias.idx.size());
    alias.aliases.resize(N);
    std::queue<int> big;
    std::queue<int> small;
    float f = N/norm;
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
        float ptot = alias.probas[s] + alias.probas[b] - 1.;
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


void print_alias(Alias& alias){
    cout << "idx: ";
    for(auto x: alias.idx){
        cout << x << " ";
    }
    cout << endl;
    cout << "aliases: ";
    for(auto x: alias.aliases){
        cout << x << " ";
    }
    cout << endl;
    cout << "probas: ";
    for(auto x: alias.probas){
        cout << x << " ";
    }
}

} // Namespace