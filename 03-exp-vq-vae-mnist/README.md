# Experiment 03 - Replication of VQ VAE with MNIST
In this experiment, we check that we can reproduce the VQ-VAE paper on the simple MNIST dataset. The replication steps are as follows:
- Learning a VQ-VAE with a fixed uniform prior
- Fitting an auto-regressive prior to the posterior to enable sampling

## Hypothesis
The original hypothesis was that a categorical prior and and posterior distribution on a learned embedding can learn to encode different modalities in an unsupervised manner.
The goal here is to replicate the result on MNIST.


## Notes
- No problem to replicate the VQ-VAE training on MNIST. I first tried with an embedding of 512 vectors, only a few vectors were used (< 100 for everyt trials) so I gave it a try with 64 vectors.
- This didn't change the results, so I went more aggressive to 32 vectors.
- At that point, all vectors were now used, the reconstruction suffered a little bit (some numbers were very shallow), and the LSTM couldn't learn anything useful in 20 epochs.
- To check the LSTM capacity, I ran it for 100 epochs using 64 vectors. The results were a little bit better but not much.

In conclusion, I found it pretty hard to fit a auto-regressive model using an LSTM. This is probably due to lack of knowledge on sequential modeling, RNN in general and auto-regressive modeling too.

### Notes about LSTM
- The trainig is pretty slow, and it seemed a hard task.
- Changing the depth of the LSTM or the depth of the decoder didn't change much.

I'm lacking the knowledge about useful tricks for LSTM training.
