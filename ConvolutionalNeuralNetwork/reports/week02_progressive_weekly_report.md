
             PROGRESSIVE WEEKLY REPORT

                WEEK 02

Group Members and Work Done

|   | Group Member                  | Work Done |

| 1 | Tindyebwa Frank (Leader)      | Zero padding implementation (Valid/Same modes) |
| 2 | Kweyamba Peter (Risk Manager) | Stride implementation and sub-sampling loop    |
| 3 | Atugonza Jeremy               | ReLU activation function module                |
| 4 | Namukasa Agatha               | Sigmoid and Tanh activation modules            |
| 5 | Mugarura Allan                | Max pooling layer implementation               |
| 6 | Mugenyi Edrine                | CNN feature integration and Week-2 demo        |
| 7 | Wanyenze Joan Natseba         | Average pooling layer implementation           |
| 8 | Evarista                      | CNN feature-extraction block builder           |

Progress

During Week 2, the group focused on extending the CNN implementation through spatial transformations, activation functions, and pooling operations. The work covered zero padding, stride and sub-sampling, ReLU, Sigmoid and Tanh activation functions, max pooling, average pooling, and integration of convolution, activation and pooling blocks.

The padding and stride operations are intended to control the spatial dimensions of tensors and the movement of convolution windows across input data. The activation functions introduce non-linearity into the network, while pooling operations reduce the spatial dimensions of feature maps. Integration work focused on combining these components into a functional CNN feature-extraction pipeline.

The project continues to follow the required four-dimensional tensor representation (N,C,H,W), using contiguous one-dimensional memory and the agreed indexing formula:


Index(n, c, h, w) = n × (C × H × W) + c × (H × W) + h × W + w


Challenges Faced

The main challenges involved ensuring compatibility between the tensor, padding, stride, activation and pooling implementations. Other challenges included maintaining correct output dimensions, handling tensor indexing, and ensuring that the individual components could be integrated and tested together. These challenges were addressed through code review, debugging, mathematical verification and testing of the implemented functions.

Conclusion

Week 2 focused on extending the foundational CNN components developed in Week 1. The group worked on padding, stride, activation functions, pooling and feature-extraction integration. These components are important for controlling spatial dimensions, introducing non-linearity and reducing feature-map sizes. The work provides a foundation for further integration, testing and development of the CNN pipeline.