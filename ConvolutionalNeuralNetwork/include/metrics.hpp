
#ifndef CNN_METRICS_HPP
#define CNN_METRICS_HPP

#include <vector>

using namespace std;

namespace cnn {

// Calculates the proportion of correct predictions.
double accuracy(const vector<int>& actual,
                const vector<int>& predicted);

// Calculates precision for one class.
double precision(const vector<int>& actual,
                 const vector<int>& predicted,
                 int class_label);

// Calculates macro-averaged precision across, all class labels found in the actual or predicted data.
double macroPrecision(const vector<int>& actual,
                      const vector<int>& predicted);

}

#endif
