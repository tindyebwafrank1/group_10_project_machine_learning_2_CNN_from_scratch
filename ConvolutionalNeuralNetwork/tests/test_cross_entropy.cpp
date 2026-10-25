#include <iostream>
#include <cassert>
#include <cmath>
#include "tensor.hpp"
#include "cross_entropy.hpp"

using namespace std;

int main() {
    cout << "[RUNNING] Verifying  Cross-Entropy Loss Module..." << endl;

    // 1. Establish structural shapes: 1 batch element, 3 classes, height 1, width 1
    size_t N = 1, C = 3, H = 1, W = 1;
    
    Tensor targets(N, C, H, W);
    
    targets(0, 0, 0, 0) = 0.0;
    targets(0, 1, 0, 0) = 1.0;
    targets(0, 2, 0, 0) = 0.0;

    CrossEntropyLoss loss_layer;

    // --- TEST 1: Near-Perfect Predictions ---
    Tensor stable_preds(N, C, H, W);
    stable_preds(0, 0, 0, 0) = 0.05;
    stable_preds(0, 1, 0, 0) = 0.90; // High probability assigned to the correct class
    stable_preds(0, 2, 0, 0) = 0.05;

    double stable_loss = loss_layer.forward(stable_preds, targets);
    double expected_stable = -log(0.90);
    assert(abs(stable_loss - expected_stable) < 1e-6);
    cout << " -> Pass: Standard stable loss calculation matches analytical value (" << stable_loss << ")." << endl;


    // --- TEST 2: Numerical Stability Bounds (The ln(0) crash protection check) ---
    Tensor risky_preds(N, C, H, W);
    risky_preds(0, 0, 0, 0) = 0.5;
    risky_preds(0, 1, 0, 0) = 0.0; // CRITICAL: 0.0 probability on correct label would trigger ln(0) -> -inf
    risky_preds(0, 2, 0, 0) = 0.5;

    double bound_loss = loss_layer.forward(risky_preds, targets);
    
    // The internal clamping must have clipped 0.0 to 1e-15, meaning loss = -log(1e-15)
    double expected_bounded = -log(1e-15);
    assert(abs(bound_loss - expected_bounded) < 1e-6);
    assert(!isnan(bound_loss) && !isinf(bound_loss));
    cout << " -> Pass: Epsilon clamping active! Protected engine from ln(0) system crash." << endl;

    cout << "\n[SUCCESS] Cross-Entropy Loss module meets all Week 3 verification metrics!" << endl;
    return 0;
}
