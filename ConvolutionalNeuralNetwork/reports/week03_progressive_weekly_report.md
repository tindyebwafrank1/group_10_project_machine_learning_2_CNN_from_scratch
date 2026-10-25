        PROGRESSIVE WEEKLY REPORT

                WEEK 03

Group Members and Work Done
| # | Group Member | Work Done |
|---|--------------|-----------|
| 1 | Tindyebwa Frank (Leader) | Flattening feature maps from 4D tensors into 2D vectors |
| 2 | Kweyamba Peter (Risk Manager) | Fully connected forward pass using `(z = Wx + b)` |
| 3 | Atugonza Jeremy | Weight and bias initialization using Xavier and He methods |
| 4 | Namukasa Agatha | Numerically stable Softmax activation layer |
| 5 | Mugarura Allan | Cross-Entropy Loss function module |
| 6 | Mugenyi Edrine | Complete forward CNN integration and final application |
| 7 | Wanyenze Joan Natseba | Prediction mechanism using argmax classification |
| 8 | Evarista | Evaluation metrics, including Accuracy and Precision |

Progress

During Week 3, the group focused on extending the CNN implementation from feature extraction to classification. The main areas of development included flattening feature maps, implementing fully connected layers, initializing weights and biases, developing the Softmax activation function, computing Cross-Entropy Loss, implementing prediction logic, and evaluating classification performance.

The flattening operation converts the feature maps produced by convolution and pooling layers into vectors suitable for the fully connected layer. The fully connected layer processes these vectors using the mathematical expression (z = Wx + b), where (W) represents the weight matrix, (x) represents the input vector, and (b) represents the bias vector.

Weight and bias initialization is intended to provide suitable starting values for network parameters. Xavier and He initialization methods are considered to help maintain appropriate activation and gradient magnitudes during training.

The Softmax activation function converts the output scores into a probability distribution over the possible classes. Numerical stability is important when calculating these probabilities. The Cross-Entropy Loss module measures the difference between predicted probabilities and the correct class labels.

The prediction mechanism selects the class with the highest output score using argmax classification. The evaluation metrics, including Accuracy and Precision, are intended to assess the performance of the classification system. Integration work focuses on connecting the feature-extraction pipeline to the classification components to form a complete forward CNN architecture.

Challenges Faced

The main challenges during Week 3 involve ensuring compatibility between the flattened feature maps and the fully connected layer, maintaining correct matrix dimensions, and initializing weights and biases appropriately. Other potential challenges include preventing numerical instability in Softmax calculations, handling logarithm and probability boundaries in Cross-Entropy Loss, and ensuring correct class selection and evaluation metric calculations.

These challenges require careful attention to mathematical specifications, consistent tensor and vector dimensions, code review, debugging, and unit testing. Integration testing is also necessary to verify that the individual modules work together correctly.

Conclusion

Week 3 focuses on extending the CNN from feature extraction to classification. The planned components include flattening, fully connected layers, weight and bias initialization, Softmax, Cross-Entropy Loss, prediction, and evaluation metrics. Together, these components form the classification stage of the network and provide a foundation for the training, backpropagation, and optimization activities planned for Week 4.

The group will continue with integration and testing to ensure that the implemented modules work together correctly and prepare the project for the final development stage.