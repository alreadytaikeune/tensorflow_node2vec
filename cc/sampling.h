#ifndef SAMPLING_H
#define SAMPLING_H

#include <vector>
#include <queue>

#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/util/guarded_philox_random.h"

using namespace tensorflow;


namespace gseq{

typedef struct Alias {
    std::vector<float> probas;
    std::vector<int> aliases;
    std::vector<int> idx;
} Alias;


void setup_alias_vectors(Alias& alias, float norm);

int sample_alias(Alias& alias, random::SimplePhilox& gen);

void print_alias(Alias& alias);

} // Namespace

#endif  // SAMPLING_H
