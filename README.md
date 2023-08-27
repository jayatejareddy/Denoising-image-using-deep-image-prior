# Image Denoising using Deep Image Prior
This repository contains an implementation of image denoising using the Deep Image Prior (DIP) technique. Deep Image Prior is a novel approach to image denoising that leverages the inherent structure of neural networks to perform denoising without the need for explicit training data. Instead, the network is "trained" on the noisy image itself.

# Deep Image Prior
Deep Image Prior is a concept introduced by Ulyanov et al. in their paper titled "Deep Image Prior" (2018). The idea behind DIP is to use the architecture of a deep neural network to regularize the solution of an inverse problem, such as image denoising. Unlike traditional denoising methods, DIP does not require a dataset of clean and noisy images for training. Instead, it exploits the network's ability to capture the image's underlying structure directly from the noisy input.

# References and Credits
* [Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2018). Deep Image Prior. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).](https://arxiv.org/abs/1711.10925)