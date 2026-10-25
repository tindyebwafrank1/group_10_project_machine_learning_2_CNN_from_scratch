
# Week 4 Report

**Name:** Kweyamba Peter (Member 2, Operations Lead)
**Task:** Fully connected layer gradients (dL/dW, dL/db, dL/dx)
**My reviewer:** Frank. **I review:** Jeremy (un-flatten gradient routing).


## 1. What I was asked to do
To train the network, we need to know how the loss changes when each weight changes. This is called backpropagation. My job was to write the backward pass for the fully connected layer. This calculates the gradients for the weights, the biases, and the input so they can be passed backward through the network.

## 2. What I made
*   `include/fully_connected_trainable.hpp` - The new class that adds the backward pass to my Week 3 class.
*   `tests/test_fully_connected_trainable.cpp` - Tests to check my maths and error handling.
*   `examples/fully_connected_training_demo.cpp` - A small demo showing the layer learning.
*   `scripts/release_check.sh` - A script to check the whole project builds and tests cleanly.

## 3. The Maths (Backpropagation)
The forward pass is `z = Wx + b`. For the backward pass, we receive `delta` (the gradient from the layers after this one). We use the chain rule to find three things:
*   **dL/dW** (weight gradient): `sum of (delta * x)`
*   **dL/db** (bias gradient): `sum of delta`
*   **dL/dx** (input gradient): `sum of (W * delta)`

*Decision:* I do not divide by the batch size (N) in my backward pass. If the loss is a mean, Frank's loss function already includes the `1/N`. This means the scaling is only done once.

*Proof:* I checked my maths using "central differences" (changing one weight slightly and seeing how the loss changes) and the numbers matched exactly.

## 4. The Class & Object-Oriented Programming
I created `FullyConnectedTrainable`. It inherits from my Week 3 class `FullyConnected`.

**How I used Encapsulation:**
*   My new gradients (`grad_weights_` and `grad_bias_`) are **private**. Other code can only read them using `const` getters, and they can only be changed by calling `backward()` or `zeroGrad()`.
*   To change the actual weights, other code must use `applyUpdate()`. This protects the layer because `applyUpdate` checks for wrong sizes or NaN values before changing anything.
*   The base class's weights are also private, so my new class cannot touch them directly. It has to use the public `setWeights()` function.

**How I used Inheritance:**
*   `FullyConnectedTrainable` uses **public inheritance**.
*   This is an "is a" relationship: a trainable layer *is a* fully connected layer with extra features.
*   It inherits all the public functions from Week 3 (like `forward`, `infer`, and the cache management) and adds the new `backward` pass. The base class doesn't need to be changed.

**How I used Polymorphism:**
*   **Compile-time polymorphism:** I did not write this in this specific class, but the project uses it via function overloading (e.g., `Stride` has different constructors) and templates (e.g., `forEachWindow`).
*   **Runtime polymorphism:** I did **not** use this. None of my functions are `virtual`. A `FullyConnectedTrainable` can be passed to a function expecting a `FullyConnected&` because of inheritance, but it doesn't use virtual pointers. I avoided virtual functions because the base class wasn't designed for polymorphic deletion (it has no virtual destructor).

## 5. Testing
The tests check:
*   The hand-calculated maths example.
*   The numerical gradient check (central differences) for weights, biases, and inputs.
*   Error handling (wrong sizes, NaN, backward before forward).
*   That the gradients sum up correctly over a batch.
*   That 20 training steps make the loss go down.
*   *Demo result:* The demo loss went from 0.6694 down to 0.0180 in 30 epochs, classifying all 8 points correctly.

## 6. Release & Risk Work
I wrote `scripts/release_check.sh` to make sure the project is ready. It checks:
1.  Official build (no warnings).
2.  Headers compile on their own.
3.  No `using namespace std;` in headers.
4.  All tests build and pass.
5.  Repo basics (README exists, no build files tracked, git is clean).

**Risks:**
*   *Loss scaling:* I need Frank to confirm his loss gradient includes the `1/N`.
*   *Optimizer:* Evarista (SGD) needs to know to use `gradWeights()` and `applyUpdate()`.
*   *Shape matching:* Jeremy and Frank need to test that flatten and un-flatten shapes match.

## 7. Reviewing Jeremy's Work
I reviewed Jeremy's un-flatten gradient routing. I checked that the gradient goes back into the original 4D shape, uses the same index formula as flatten, rejects wrong sizes, and works with my `dx` output.
*Result:* (Approved / Changes asked)

## 8. Limitations
*   `backward()` can only be called after `forward()` on the same batch.
*   Gradients are replaced, not added up.
*   The code uses plain loops, so very large layers might be slow.
*   Never delete a `FullyConnectedTrainable` through a `FullyConnected` pointer (no virtual destructor).

## References
-  I. Goodfellow, Y. Bengio and A. Courville, *Deep Learning*. MIT Press, 2016.
-  D. E. Rumelhart, G. E. Hinton and R. J. Williams, "Learning representations by back-propagating errors," *Nature*, 1986.
-  C. M. Bishop, *Pattern Recognition and Machine Learning*. Springer, 2006.
-  B. Stroustrup, *The C++ Programming Language*, 4th ed. Addison-Wesley, 2013.
-  S. B. Lippman, J. Lajoie and B. E. Moo, *C++ Primer*, 5th ed. Addison-Wesley, 2013.