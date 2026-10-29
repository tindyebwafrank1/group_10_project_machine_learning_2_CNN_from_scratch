# PROGRESSIVE WEEKLY REPORT

## WEEK 04

### Group Members and Work Done

| # | Group Member | Work Done |
|---|---|---|
| 1 | Tindyebwa Frank (Leader) | Implemented loss gradient computation and the Softmax backpropagation derivative. |
| 2 | Kweyamba Peter (Risk Manager) | Worked on fully connected layer gradients, including gradients for weights, biases, and inputs. |
| 3 | Atugonza Jeremy | Worked on flattening layer gradient routing to propagate gradients back to the original feature maps. |
| 4 | Namukasa Agatha | Implemented activation gradients for ReLU, Sigmoid, and Tanh functions. |
| 5 | Mugarura Allan | Worked on Max Pooling and Average Pooling backward passes, including gradient distribution. |
| 6 | Mugenyi Edrine | Worked on the Conv2D backward pass, including kernel gradients and input spatial gradients. |
| 7 | Evarista | Worked on the Stochastic Gradient Descent (SGD) optimizer step for updating model parameters. |
| 8 | Wanyenze Joan Natseba | Worked on the mini-batch training loop, epoch execution, integration, and final release package. |

---

### Progress

During Week 4, the group focused on extending the CNN implementation from forward propagation to backpropagation and optimization. The main areas of development included loss gradient computation, fully connected layer gradients, flattening gradient routing, activation gradients, pooling backward passes, convolution backward propagation, parameter optimization, and mini-batch training.

Loss gradient computation provides the starting point for backpropagation by determining how the loss changes with respect to the network's output. The Softmax backpropagation derivative supports the propagation of these gradients through the classification stage. The fully connected layer gradients determine how the loss changes with respect to the weights, biases, and input features.

The flattening layer gradient routing operation transfers gradients from the flattened feature vectors back to the original feature maps. Activation gradients for ReLU, Sigmoid, and Tanh enable the network to propagate gradients through its nonlinear activation functions.

The pooling backward passes distribute gradients to the appropriate input positions. In Max Pooling, the gradient is routed to the input position that produced the maximum value in the forward pass. In Average Pooling, the gradient is distributed across the input elements covered by the pooling window.

The Conv2D backward pass computes gradients with respect to the convolution kernels and input feature maps. These gradients are necessary for updating convolution parameters and propagating learning signals to earlier layers. The Stochastic Gradient Descent (SGD) optimizer updates the model parameters using the calculated gradients.

The mini-batch training loop coordinates the training process by dividing training data into batches, performing forward propagation, calculating the loss, executing backpropagation, and updating parameters. Epoch execution repeats this process over the training dataset. The final integration and release activities bring the individual components together into the project deliverable.

---

### Challenges Faced

The main challenges during Week 4 involve ensuring that gradients have the correct dimensions and are routed to the appropriate layers. Backpropagation requires careful handling of tensor shapes, matrix operations, activation derivatives, and pooling indices.

Additional challenges include calculating convolution kernel and input gradients correctly, ensuring that the optimizer updates parameters using the correct gradients, and maintaining consistent interfaces between modules developed by different group members. Integrating all the backward-pass components into a complete training loop also requires careful testing to identify errors that may not appear when individual modules are tested separately.

These challenges require checking mathematical expressions, validating tensor dimensions, reviewing module interfaces, debugging compilation errors, and performing unit and integration tests. The final release process also requires confirming that the project builds correctly and that the available tests and examples run as expected.

---

### Conclusion

Week 4 focuses on completing the training stage of the CNN by implementing backpropagation and optimization. The assigned components cover loss gradients, fully connected layer gradients, flattening gradient routing, activation derivatives, pooling backward passes, convolution backward propagation, SGD optimization, and mini-batch training.

Together, these components provide the mechanisms required for the network to calculate gradients, propagate them through its layers, and update its parameters during training. The integration of these modules forms the final development stage of the project.

The group will focus on final integration, testing, resolving compatibility issues, and preparing the release package to ensure that the CNN implementation is ready for submission.