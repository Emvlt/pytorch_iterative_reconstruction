# pytorch_iterative_reconstruction
Repository for Pytorch-only iterative reconstruction framework

Pytorch Iterative Reconstruction is a framework to solve an inverse problem (here, 2D XCT) using variational regularization in Pytorch.

It relies on grid sampling to make the rotation a differentiable operation, and then by minimising a loss function in the projection space.

