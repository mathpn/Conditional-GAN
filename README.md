# Conditional-GAN

[Generative Adversarial Networks (GANs)](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf) are a popular type of neural network that generates new unseen data from a randomly sampled input vector.

Their results can look very realistic, but, realistically, it's not feasible to achieve photorealism without GPU training.
Thus, the goal here is simply to experiment with GANs using the [102 flowers dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/). All training routines were done using 64x64 pixel images on a CPU.

## Vanilla GAN

We start first with a vanilla GAN. Its generator takes a 64-dimensional vector sampled from a gaussian distribution. The Adam optimizer is used throughout with a batch size of 128. GANs are known to present a few [training issues](https://developers.google.com/machine-learning/gan/problems). I've observed partial mode collapse, which was tackled by adding a small gaussian noise to all discriminator inputs.


## Conditional GAN

Since our dataset is labeled, we may use [Conditional GANs](https://arxiv.org/abs/1411.1784). The idea is simple: we have two embeddings for the 102 classes, one for the generator and another for the discriminator. We present a randomly sampled vector concatenated with the embedding vector to the generator. The discriminator concatenates its embedding vector with the output of its convolutional layers just before the last fully-connected layer.

The idea is to provide a learnable vector to represent the prototypical image of each class, helping both the generator and the discriminator to get better at their jobs. This might potentially lead to more photorealistic images.

One issue was that the generator relied too heavily on the embedding vector, which resulted in partial mode collapse: all outputs for a given label looked almost identical. To prevent this, a 0.5 probability dropout was added to the generator embedding.
