
#ifndef CNN_CROSS_ENTROPY_HPP
#define CNN_CROSS_ENTROPY_HPP

#include <vector>

using namespace std;

// Calculates categorical cross-entropy for one prediction.
// probabilities and targets must have the same length.
double crossEntropyLoss(
    const vector<double>& probabilities,
    const vector<double>& targets,
    double epsilon = 1e-12
);

// Calculates the average cross-entropy loss across a batch.
// Each row represents one sample and its class probabilities/targets.
double crossEntropyLossBatch(
    const vector<vector<double>>& probabilities,
    const vector<vector<double>>& targets,
    double epsilon = 1e-12
);

#endif
