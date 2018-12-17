#include <iostream>
#include <vector>
#include <cassert>
#include <queue>
#include <random>
static std::random_device rd; // random device engine, usually based on /dev/random on UNIX-like systems
static std::mt19937 gen(rd());

typedef struct Alias {
  std::vector<double> probas;
  std::vector<int> aliases;
  std::vector<int> idx;
  std::uniform_int_distribution<int> uid;
  std::uniform_real_distribution<double> dis;
} Alias;


void setup_alias_vectors(Alias& alias){
    int N = alias.probas.size();
    assert(alias.probas.size() == alias.idx.size());
    alias.aliases.resize(N);
    std::queue<int> big;
    std::queue<int> small;
    std::vector<double> alias_probas(N);
    for(int i=0; i<N; i++){
        std::cout << alias.probas[i] << " ";
        alias_probas[i] = alias.probas[i]*N;
        if(alias.probas[i]*N < 1.)
          small.push(i);
        else
          big.push(i);
    }
    std::cout << "\n";
    while(!(big.empty() || small.empty())){
        std::cout << "size " << big.size() << " " << small.size() << std::endl;
        int s = small.front(); small.pop();
        int b = big.front(); big.pop();
        alias.aliases[s] = b;
        double ptot = alias_probas[s]*N + alias_probas[b]*N - 1.;
        alias_probas[b] = ptot;
        if(ptot < 1.){
          small.push(b);
        }
        else{
          big.push(b);
        }
    }
    // assert(big.empty() && small.empty());
    for(int i = 0; i<N; i++){
        std::cout << alias_probas[i] << " ";
        alias.probas[i] = alias_probas[i];
    }
    std::cout << "\n";
    
    alias.uid = std::uniform_int_distribution<int>(0, alias.probas.size()-1);
    alias.dis = std::uniform_real_distribution<double>(0, 1);
}


int sample_alias(Alias& alias){
    int N = alias.probas.size();
    int v = alias.uid(gen);
    double x = alias.dis(gen);
    if(x < alias.probas[v]){
        return alias.idx[v];
    }
    return alias.idx[alias.aliases[v]];
}

int main(){
    Alias a;
    a.probas.push_back(1.); a.probas.push_back(0.); a.probas.push_back(0.);
    a.idx.push_back(1); a.idx.push_back(0); a.idx.push_back(2);
    setup_alias_vectors(a);
    int res[3]; res[0]=0; res[1]=0; res[2]=0;
    for(int t=0; t<1000; t++){
        int i = sample_alias(a);
        res[i]++;
    }

    std::cout << res[0] << " " << res[1] << " " << res[2] << std::endl;
    return 0;
}