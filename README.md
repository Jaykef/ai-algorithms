# AI Algorithms

This repo is a work in progress containing first-principle implementations of groundbreaking AI algorithms using a wide range of deep learning frameworks. Each implementation is accompanied by its supporting research paper(s). The goal is to provide comprehensive educational resources for understanding and implementing foundational AI algorithms from scratch.

## Implementations
- [mnist_self_compressing_nns](https://github.com/Jaykef/ai-algorithms/blob/main/mnist_self_compression.ipynb) - Pytorch implementation of ["Self-compressing Neural Networks"](https://arxiv.org/pdf/2301.13142). The paper shows dynamic neural network compression during training - reduced size of weight, activation tensors and bits required to represent weights.
- [mnist_ijepa](https://github.com/Jaykef/ai-algorithms/blob/main/mnist_ijepa.ipynb) - Simplified image-based implementation of JEPA (Joint-Embedding Predictive Architecture) - an alternative to auto-regressive LLM architectures pioneered by Prof. Yann LeCun. [I-JEPA](https://arxiv.org/pdf/2301.08243.pdf) predicts image segment representations (Target) based on representations of other segments within the same image (Context).
- [nns_are_decision_trees](https://github.com/Jaykef/ai-algorithms/blob/main/nns_are%20decision_trees.ipynb) - Simplified implementation of “Neural Networks are Decision Trees”. Showing that any neural network with any activation function can be represented as a decision tree. Since decision trees are inherently interpretable, their equivalence helps us understand how the network makes decisions.
- [mnist_the_forward_forward_algorithm](https://github.com/Jaykef/ai-algorithms/blob/main/mnist_the_forward_forward_algorithm.ipynb) - Implements the [Forward-Forward Algorithm](https://arxiv.org/abs/2212.13345) proposed by AI godfather Geoffrey Hinton. The algorithm replaces the forward and backward passes in backpropagation with two forward passes on different data with opposite objectives. The positive pass uses real data and adjusts weights to increase goodness in every hidden layer. The negative pass uses "negative data" and decreases goodness.
- [sigmoid_attention](https://github.com/Jaykef/ai-algorithms/blob/main/sigmoid_attn.ipynb) - Implements newly introduced [Sigmoid Self-Attention](https://arxiv.org/abs/2409.04431) by Apple.
- [DIFF_Transformer](https://github.com/Jaykef/ai-algorithms/blob/main/DIFF_Transformer.ipynb) - Lightweight implementation of newly introduced “Differential Transformer”: Proposes differential attention mechanism which computes attention scores as a difference between two separate softmax attention maps thereby reducing noise in attention blocks. [Paper](https://arxiv.org/pdf/2410.05258) by microsoft.
- [triton_nanoGPT.ipynb](https://github.com/Jaykef/ai-algorithms/blob/main/triton_nanoGPT.ipynb) - Implements custom triton kernels for training Karpthy's nanoGPT (more improvements needed).
- [generating_texts_with_rnns.ipynb](https://github.com/Jaykef/ai-algorithms/blob/main/generating_texts_with_rnns.ipynb) - Implements ["Generating Text with Recurrent Neural Networks"](https://icml.cc/2011/papers/524_icmlpaper.pdf) - trains a character-level multiplicative recurrent neural network model (~250k params) for 1000 epochs on [2pac](https://github.com/Jaykef/ai-algorithms/blob/main/data/tupac.txt)'s "Hit 'em Up" lol, sample was fun:)
- [deep_pcr.ipynb](https://github.com/Jaykef/ai-algorithms/blob/main/deep_pcr.ipynb) - Implements ["DeepPCR: Parallelizing Sequential Operations in Neural Networks"](https://machinelearning.apple.com/research/deeppcr) - a novel algorithm which parallelizes typically sequential operations in order to speed up inference and training of neural networks.
- [seq2seq_with_nns](https://github.com/Jaykef/ai-algorithms/blob/main/seq2seq.ipynb) - Lightweight implementation of the seminal paper “Sequence to Sequence Learning with Neural Networks”. Built, trained and eval a 2 layer deep seq2seq LSTM-based model (~10M params) on German-English corpus of Multi30K dataset. In honor of Ilya sutskever et al for winning this year’s NeurIPs Test of Time paper award.
- [discrete_flow_matching](https://github.com/Jaykef/ai-algorithms/blob/main/dfm.ipynb) - Implements from first-principles a discrete flow matching (DFM) model for code generation. In particular we trained a small sized 2d dfm model on two variations of code for binary search. DFM is a non-autoregressive generative modeling framework recently introduced in this <a href="https://arxiv.org/pdf/2407.15595">paper</a> by meta.
- [byte_latent_transformer](https://github.com/Jaykef/ai-algorithms/blob/main/byte_latent_transformer.ipynb) - Here we implement a charcter-level BLT (Byte Latent Transformer) model from scratch under 500 lines of code. The Byte Latent Transformer architecture is a tokenizer-free architecture that learns from raw byte data, recently introduced in this [paper](https://dl.fbaipublicfiles.com/blt/BLT__Patches_Scale_Better_Than_Tokens.pdf) by meta.

- [llm_knowledge_distillation](https://github.com/Jaykef/ai-algorithms/blob/main/byte_latent_transformer.ipynb) - a minimal single script implementation of knowledge distillation in LLMs. Knowledge distillation uses a large pretrained “teacher” model to guide the training of a smaller “student” model so that the student mimics the teacher’s core capabilities but at reduced computational cost. In this implementation, we use GPT-2 124M as student model and GPT-2 Medium 340M as teacher via reverse Kullback-Leibler (KL) divergence. Reference Papers: https://arxiv.org/abs/1503.02531, https://arxiv.org/abs/2306.08543 
