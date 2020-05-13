# ML tricks

Notes on ML tricks for ensuring faster convergence to global minimum.

[On the Convergence of Adam and Adagrad](https://arxiv.org/pdf/2003.02395.pdf)
A main practical takeaway, increasing the exponential decay factor is as critical as decreasing the learning rate for converging to a critical point. Our analysis also highlights a link between Adam and a finite-horizon version of Adam: for fixed N,taking α = 1/√N and β2 = 1 − 1/N for Adam gives the same convergence bound as Adagrad

[ReZero](https://arxiv.org/pdf/2003.04887.pdf)
The idea is simple: ReZero initializes each layer to perform the identity operation. For each layer, we introduce a residual connection for the input signal x and one trainable parameter α that modulates the non-trivial transformation of the layer F(x),

[Statistical Adaptive Stochastic Gradient Methods](https://github.com/microsoft/statopt)
Automatically find a good learning rate.
!! Need to be tested and compared to the good old Adam optimizer

[Evolving Normalization-Activation Layers](https://arxiv.org/pdf/2004.02967.pdf)
EvoNorm-S0 normalisation layer + activation function : \frac{x * \sigma(v_1 * x))}{ \sqrt{std_{h,w,c/g}^2(x) + \epsilon} * \gamma + \beta }
v_1, \gamma and \beta are learnable weights

[Do We Need Zero Training Loss After Achieving Zero Training Error?](https://arxiv.org/abs/2002.08709)
Ensure you can't reach 0-training loss while reaching 0-training error. This leads the NN to random walk and reach "flat" minima.

[The large learning rate phase of deep learning-the catapult mechanism](https://arxiv.org/pdf/2003.02218.pdf)
A way to initialize your learning rate. Compute the curvature at initilization and use to define the critical learning rate. (This holds for SGD, large width, ReLU networks, MSE loss)


## Empiric Scaling formula
[A constructive prediction of the generalization error across scales](https://openreview.net/pdf?id=ryenvpEKDr)

[Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf)


## Weight normalization
Micro-batch training (1 datum per batch) can pose problems for training. [Weight standardization](https://github.com/joe-siyuan-qiao/WeightStandardization) used with [Group Normalization](https://arxiv.org/abs/1803.08494) solved those problems
(the statistics of the weights are computer over c_in x w_k x h_k)


## Engineering
[Data echoing](https://ai.googleblog.com/2020/05/speeding-up-neural-network-training.html) to alleviate your input data pipeline latency.