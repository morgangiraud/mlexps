# Visual semantic grounding

Overleaf draft: https://www.overleaf.com/project/5e82e9c295a26000019a5ea5

This repo contains a bunch of ML experiments exploring visual semantic grouding.

It is a journey into quantized representation. I'm exploring it at many levels.
What could a quantized representation brings to the table?
What are the advantages?
- discrete representations are potentially a more natural fit for many of the modalities (languages, speech, complex reasoning, planning and predictive learning).
- Anywhere there is a notion of sequences, discrete timestep and representation is interesting
- Autoregressive models have been developed for modelling distributions over discrete variables
-> check the argument of go-explore
What are the potential limitations?
Are lottey tickets a kind of weight quantization ?
Does the brain quantize information ? Does it induce a framerate?

But before going into quantization the first step is to look at representation. Representation learning is a vast area with a lot of ideas. I'm not going to explore this field, instead I'll start with a solution which I find well-motivated : VAE.

Why VAE? Because they are inspired from 2 ideas that I find well-motivated:
- Variationnal inference and its subsequent derivation, black box variationnal inference. A solution motivated by the need to approximate an intractable bayesian posterior. It is an alternative Monte-Carlo solutions and looks to hold a lot of promise to learn probability distribution
- The second is autoencoder which is an unsupervised learning technique. Due to its unsupervised nature, The learnt representation is not bias to any goals (outside of compression if using the reconstruction loss) which has advantages and inconvenient.
Overall it is also an easy algorithm to pull off and pretty straight forward. It can also be easily adptated to different loss to incorporate notions of predictive coding for example. One can also bias the compression by biasing the training set.

#### Exp 1
Experiment 1 is about BBVI ([Black-Box Variational Inference](http://www.cs.columbia.edu/~blei/papers/RanganathGerrishBlei2014.pdf) VI where you can't have a closed form and use stochastic optimization like SGD which uses samples from a distribution)

Notes:
- It is also interesting to note that VI is at the heart of active inference and all of Karl Friston Work. Even though this line of work is not yet validated, I feel like it's a promising direction for RL and representation learning.
- Also, VI is used to infer latent variables, latent vriables can be interpreted as belief about how the world works and beliefs are at the heart of human mind. What is the link between belief, ideas and language in term of representation?

From there we will look at quantization of the representation. There I will go directly with one of the work which maps perfectly with my direction: [VQVAE](https://arxiv.org/pdf/1711.00937.pdf).

My main interest in quantized representation is the possibility to learn a representation based on symbols (which are represented into an embedding). The idea is that, it might ease the capacity to map it to a word embedding later on, or a sentence embedding. (Any way I strongly believe into a deeply recursive embedding: letters, words, sentence, paragraph, complete text should be in the same space)

#### Exp 2 and 3
Those experiements reproduces the VQVAE paper and make sure that we indeed can learn a VQVAE but also fit an autoregressive model to the prior disteribution.

From there, I started to wonder how to use quality-diversity and illumination algorithms with my quest. A [paper](https://arxiv.org/pdf/2003.04389.pdf) came to the rescue by showing that I could create a biased training set of high performing solutions to a given problem for my VQVAE.

The goal is to find elites and compress their representation at the same time. The archived elites represent the biased training set for the VQVAE. This allow the VQVAE to learng common features contained only in elites solutions.
The interesting point of such a compressed representation is that one can then search this space to find new solutions to the current problem (directing the search) and also be used to solve a new problem which should share some properties of the elites. In my example, all the elites has a low joint variance so I could search a solution to new problem which needs low joint variance, at least that is the intuition.

#### Exp 4
I replicated the Automating Representation Discovery with MAP-Elites paper. Everything went very smoothly and it's pretty nice to mix gradient based and gradient free method to solve a problem.
One curious thing (it would need more exploration to be clarified) is that all representations in this experiment end up by aligning the embedding vectors. Potentially, there is a bias in VQVAE.


Equipped with my capacity to learn an "interesting" quantized representation for a given task I believe I'm ready to move on to incorporate the notion of sequences. To do so, I focus my inspiration on world models in general.
Obvisouly, I'm curious about the paper World Models by Ha & Schmidhubber and MuZero from deepmind.

So far I wonder how words are related to world models. We use words to communicate, it seems that we use sequences because we are limited by the quantity of information we can share at a given timestep. I'm writing a sequence because I can't just write one word containing everything I want to write, probably because this woudkl require so many words that it quickly become intractable to our mind so we use a composable systems. Everything in the end, is an object, a idea quantum. If that is the case, then exploring quantized representation is definitely well-motivated. Can ideas be continuous? It doesn't feel like it. It doesn't feel like there is some meaning about interpolating ideas.
It feels like GAN, where even though it is quite intertaining there is no reason to believe there is a continous manifold of faces in which you can interpolate and so, while interpolating, you will find aberration. Yes but, the discretization of the face manifold could be very "dense" (I know it doesn't mean much) which is why the aberration we see are not too weird.
Also sequence might be needed jsut because our brain needs to do some kind of temporal sampling. At potentially learned timesteps, the brain needs to integrate information from the sense and his inner compute to its own state.
I also have som intuition about the fact that if every interesting point is an attractor in its field, then any noisy version of that point will converge to the denoised point as long as points are sufficiently "far apart".

Enough divergence!
Let's move on to world models and replicate the Ha and Schmidhubber paper.