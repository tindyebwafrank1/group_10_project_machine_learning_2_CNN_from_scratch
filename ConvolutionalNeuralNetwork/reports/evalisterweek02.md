  ## Week 2 Progress Report: Ssempewo Evalister Entry

  ## Task Assigned: Feature Block Module (Combining Convolution, Activation and Pooling)

  ## Completed
- Designed the structure for the header-only include/feature_block.hpp module to combine convolution, activation and pooling into one reusable feature-extraction block.
- Defined the `FeatureBlock` interface to support configurable input channels, output channels, kernel size, stride, padding, activation function and random seed.
- Planned the forward-pass sequence: Conv2D → Activation → Pooling.
- Incorporated the specified He weight-initialization approach, using a fixed random seed for reproducible initialization.
- Defined the `set_kernel(k)` interface for replacing filters with the required shape `{out_c, in_c, k, k}`.
- Designed the `output_shape(H, W)` function to calculate output dimensions without processing image data.
- Planned shape validation to detect invalid dimensions and prevent out-of-range tensor access.
- Defined the `FeatureExtractor` builder to chain multiple feature blocks, infer input channels from preceding blocks and calculate final feature-map dimensions.

  ## In Progress
- Integrating the feature-block module with the project's existing Tensor, Conv2D, activation and pooling interfaces.
- Coordinating with teammates to ensure consistent header paths, namespaces and function signatures.
- Preparing integration tests for individual blocks and stacked feature extractors.

  ## Challenges/Blockers
- Ensuring compatibility between the feature-block interface and the existing implementations of Tensor, convolution, activation and pooling.
- Handling output-dimension calculations correctly when stride, padding and pooling are applied.
- Confirming that convolution padding and pooling interfaces match the project's final merged versions.

  ## Next Week
- Integrating the feature-extraction blocks into the CNN forward pipeline.
- Testing multiple stacked blocks with different channel counts and kernel configurations.
- Preparing the feature extractor for integration with the subsequent CNN layers.

  ## AI Use
- Tool:   ChatGPT
- Purpose: Assisted with planning the feature-block module structure, forward-pass sequence, output-shape calculations and integration requirements for the CNN project.