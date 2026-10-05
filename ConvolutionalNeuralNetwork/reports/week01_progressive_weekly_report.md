 PROGRESSIVE WEEKLY REPORT

 WEEK 01

--Group Member                --Work done

1 Tindyebwa Frank(Leader) |   Image file loading engine                      
2 Kweyamba Peter          |   Image preprocessing                                             
3 Atugonza Jeremy         |   Image representation data structure            
4 Namukasa Agatha         |   Tensor class implementation                         
5 Mugarura Allan          |   Kernel and filter initialization                             
6 Mugenyi Edrine          |   Convolution pipeline integration & Week-1 demo 

--Progress

The group established the initial interfaces and data structures required for the CNN implementation. Image loading and image representation were successfully implemented and tested. Work was also started on preprocessing, tensors, kernels, filters, and some convolution operations.

The project follows the required 4D tensor representation (N, C, H, W) using contiguous 1D memory, with the agreed indexing approach:

Index(n,c,h,w) = n × (C × H × W) + c × (H × W) + h × W + w

--Challenges faced

The main challenges involved coordinating the different components and ensuring that the interfaces between members' work were compatible. These were addressed through testing and discussion among the group members.

--Conclusion

Good progress was made during Week 1. The foundational image handling and representation components were successfully implemented, while the remaining members progressed with the preprocessing, tensor, kernel, and convolution components. The group is now ready to continue with integration and further testing.
