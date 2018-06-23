# NAG-pytorch
Network for Adversary Generation

### Unofficial implementation of CVPR 2018 paper: "NAG: Network for Adversary Generation" in PyTorch.

### Abstract

> Adversarial perturbations can pose a serious threat for deploying machine learning systems. Recent works have
> shown existence of image-agnostic perturbations that can fool classifiers over most natural images. Existing methods
> present optimization approaches that solve for a fooling objective with an imperceptibility constraint to craft the > perturbations. However, for a given classifier, they generate one perturbation at a time, which is a single instance from
> the manifold of adversarial perturbations. Also, in order to build robust models, it is essential to explore the manifold
> of adversarial perturbations. In this paper, we propose for the first time, a generative approach to model the distribution
> of adversarial perturbations. The architecture of the proposed model is inspired from that of GANs and is
> trained using fooling and diversity objectives. Our trained generator network attempts to capture the distribution of
> adversarial perturbations for a given classifier and readily generates a wide variety of such perturbations. Our experimental
> evaluation demonstrates that perturbations crafted by our model (i) achieve state-of-the-art fooling rates, (ii)
> exhibit wide variety and (iii) deliver excellent cross model generalizability. Our work can be deemed as an important
> step in the process of inferring about the complex manifolds of adversarial perturbations.

### Architecture

![Architecture](https://github.com/val-iisc/nag/blob/master/extras/nag.png)

