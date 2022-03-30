# Discriminative Representation Learning for Rigid Registration of diffusion-weighted MR Images
![image](fig/image.png)

## Table of Contents

- [Abstract](#security)


## Abstract

Head motion correction is a critical step in the current mainstream preprocessing pipelines for diffusion-weighted MRI (dMRI), and it can be considered a rigid registration problem for diffusion image volumes of varied b-values and gradient directions, differing drastically in their appearances. In addition, the dMRI images often contain severe susceptibility and eddy current induced distortions. Hence, it is difficult to define an optimal registration cost function for all volumes of a dMRI image. This problem can be alleviated by alternatingly and iteratively performing distortion correction and rigid registration with template learning, e.g. in the HCP diffusion preprocessing pipeline, though at a large cost of computation time. In this work, we propose a deep learning based method for learning the discriminative features for dMRI image registration. The registered images can then be fed to susceptibility distortion correction and eddy current distortion correction algorithms to produce the final preprocessed images in one shot, which significantly reduce the preprocessing time cost. By adopting a training loss defined by the feature-wise difference, the learned network produces image features that maximizes the feature-wise differences for misaligned samples while minimizing the differences for well aligned samples. The experimental results show that our method outperforms the training alignment examples for dMRI image preprocessing, and achieves similar results of the HCP pipeline at significantly reduced time cost. 


