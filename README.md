# AI Algorithms

This repo is a work in progress containing first-principle implementations of various AI algorithms using a wide range of deep learning frameworks, accompanied by relevant research papers. The goal is to provide comprehensive educational resources for understanding and implementing foundational AI concepts from scratch.

## Implementations
- [mnist_self_compression](https://github.com/Jaykef/ai-algorithms/blob/main/mnist_self_compression.ipynb) - Pytorch implementation of ["Self-compressing Neural Networks"](https://arxiv.org/pdf/2301.13142). The paper shows dynamic neural network compression during training - reduced size of weight, activation tensors and bits required to represent weights.
- [mnist_ijepa](https://github.com/Jaykef/ai-algorithms/blob/main/mnist_ijepa.ipynb) - Simplified image-based implementation of JEPA (Joint-Embedding Predictive Architecture) - an alternative to auto-regressive LLM architectures pioneered by Prof. Yann LeCun. [I-JEPA](https://arxiv.org/pdf/2301.08243.pdf) predicts image segment representations (Target) based on representations of other segments within the same image (Context).
