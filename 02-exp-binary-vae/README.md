# Experiment 02 - Binary VAE

In this experiment, we explore if we can learn the latent distribution of binary data.

The data is generated as follow:
- We draw a number of points N from a mixture of 3 2D Gaussian
- We project those points in a 100 dimensionnal space
- We apply a cosine non-linearity
- Finally, we make those value binay (1 if x > 0 else 0)

## Hypothesis
A DNN VAE can recover the original parameters of the 2D Gaussian distribution from which the data point is sampled.

## Result
No.
In this experiment the AVE did not recover the original distribution in the latent space.
Yet, it still was able to separate the data by class in a unsupervised manner.

### Notes
Le VAE binaire en dimension 100 fonctionne, dans le sens qu'il arrive bien à séparer les données dans l'espace latent :
- Les données sont positionnées relativement les unes des autres de manières similaires au données originales
- Cette séparation se fait totalement de manière non-supervisé, à aucun moment le VAE utilise de labels
- On observant les distributions moyennes par classe, on voit qu'elles sont bien espacées.
- Cependant, il n'obtient pas la mixture de Gaussienne originale. Cela s'explique pour moi dans le sens, que pour reconstruire les données à partir de la représentation latente. Il suffit de bien la séparer.

### Learning rate scheduling
Some papers have been suggesting that the very first iterations of SGD learning is important and a high learning rate should be used so I've been exploring the impact on this toy task.

With a constant 3e-4 lr, we seems to converge to a "continuous ring" of latent representation.

With a decreasing learning rate starting from 3e-2, we converge to a more clearly separated clusters losing the "continuity".

### Batch size
Nothing special here, smaller batch size, more noise. The added noise doesn't seem to improve much anything in this task. Probably too much a toy task.