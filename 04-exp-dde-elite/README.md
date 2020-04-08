# Experiment 04 - Replication of
In this experiment, we check that we can reproduce the representation search using map-elites to bias the encoding of the VQ VAE towards high performing solutions.
- Illuminating a map a solutions
- Learn a quantized encoding

Paper: https://arxiv.org/pdf/2003.04389.pdf

## Hypothesis
The original hypothesis here is that one can use an autoencoder on a specified subset of solutions to bias the VAE towards high performming solutions.

We can verify this hypothesis by reusing the quantized embedding and the decoder as genetic-phenotype mapping to search for high performing solutions in a related problem.

## Notes
- No problem to replicate the paper using a quantized VAE.
- Interesting fact, when using only 2d vectors, the VQ vae learns only 2 directions for all those vectors, moreover they are equally distributed along those 2 dimensions.

### Notes about ReZero
It definitely stabilize the VQVAE training, ensuring the perplexity to go high for every run. On the other hand, without it, you can have some lucky start which greatly accelerate the discoveries of elites solutions.


### Questions
- I wonder if I missed something about the algorithm that could explain why all the vectors aligned in only 2 dimensions very cleanly?
- Now that I can learn a biased quantized embedding for high performing solutions. How to apply it to my main question of improving state predictions in a multiplayer game? A natural step would to start from a pretrained word embedding and use it as an init in a VQVAE to encode interesting part of the state. The is probably a way to map the encoding of a game description with the current visual encoding of the state. I have some difficulty to handle the notion of sequence (for the sentence) and a one-step projections of the game-step into the vectors used in the embedding. Maybe I should instead use hierarchical embedding?