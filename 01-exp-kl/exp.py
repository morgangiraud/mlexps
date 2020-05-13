###
# Experiment to check that we can learn a distribution by minimizing
# a Riemannienne approximation of the KL divergence.
# Note that the KL divergence behaves nicely only on the union of the
# distributions
# support.
###
import os
import argparse

import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as manimation
import seaborn as sns
matplotlib.use("Agg")
import neptune

cfd = os.path.dirname(os.path.realpath(__file__))
results_dir = os.path.join(cfd, 'results')

ap = argparse.ArgumentParser()
ap.add_argument("--type", required=False, default="rsum", help="Spanning strategy")
ap.add_argument("--span", required=False, default="both", help="Spanning strategy")
args = vars(ap.parse_args())

# Some matplotlib tooling
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(
    title='KL loss fitting',
    artist='Morgan Giraud',
    comment='Learning a distribution using the KL divergence'
)
writer = FFMpegWriter(fps=15, metadata=metadata, bitrate=1800)
fig = plt.figure()


def kl_loss_rsum(dist_q, dist_p, data, delta):
    q_data = torch.pow((data - dist_q.mean) / dist_q.stddev, 2)
    p_data = torch.pow((data - dist_p.mean) / dist_p.stddev, 2)

    Z_q = (dist_q.stddev * np.sqrt(2 * np.pi))
    q = 1 / Z_q * torch.exp(-0.5 * q_data)

    log_dist_ratio = torch.log(dist_p.stddev / dist_q.stddev) - 0.5 * (q_data - p_data)

    return torch.sum(q * log_dist_ratio * delta)


def kl_loss_grad_estimator(dist_q, dist_p):
    # we take advantage of the reparametrization trick to compute
    # an unbiased estimate of the gradient
    # Which actually requires only one datapoint sample from q per iteration :D
    # To reduce the noise, we samples 10 points
    e = np.random.standard_normal([10])
    data = dist_q.mean + dist_q.stddev * torch.tensor(e)

    p_data = torch.pow((data - dist_p.mean) / dist_p.stddev, 2)

    # The full KL loss is equal to
    # log(p.stddev) - log(q.stddev) - 0.5 * (q_data - p_data)
    # Following is the reduction when doing minimization
    log_dist_ratio = -torch.log(dist_q.stddev) + 0.5 * p_data

    return torch.mean(log_dist_ratio)


# Seed
torch.random.manual_seed(1)
np.random.seed(10)

# Let's define the true distribution: p(x) ~ N(mu, sigma)
mu = 1.
sigma = .3
dist_p = torch.distributions.Normal(mu, sigma)

# Let's define the parametrised distribution: dist_q
w_mu = torch.tensor(5., requires_grad=True)
w_sigma = torch.tensor(.8, requires_grad=True)
dist_q = torch.distributions.Normal(w_mu, w_sigma)

# Number of data points
N = 500
delta = 6 / N

# The KL divergence contains an integral for which we do not have a closed form
# So we need to approximate it
# The simplest idea is to approximate it using a riemannien sum

# KL loss approximated using a Riemanienne sum
# We sample N evenly spaced points in [a, b]
# the delta is then (b-a)/N
# A good interval is one spanning both distributions support
# In our case of 2 gaussians, if you:
# - span only q, You will "leave" your original support but
# then you don't have anymore information to keep going
# - span only p, you will get very small gradients and very slow convergence
if args["type"] == "rsum":
    dist_q_mean = dist_q.mean.item()
    dist_q_stdddev = dist_q.stddev.item()
    dist_p_mean = dist_p.mean.item()
    dist_p_stdddev = dist_p.stddev.item()
    if args["span"] == "q":
        video_path = os.path.join(results_dir, "kl_loss_fitting-span_q.mp4")
        data = torch.tensor(
            np.linspace(  # data spanning only q
                dist_q_mean - 4 * dist_q_stdddev,
                dist_q_mean + 4 * dist_q_stdddev,
                N
            ))
        lr = 1e-4
        # Fast convergence to the border of the interval
        training_steps = 500
    elif args["span"] == "p":
        video_path = os.path.join(results_dir, "kl_loss_fitting-span_p.mp4")
        data = torch.tensor(
            np.linspace(  # data spanning only p
                dist_p_mean - 4 * dist_p_stdddev,
                dist_p_mean + 4 * dist_p_stdddev,
                N
            ))
        lr = 1e-3
        # very slow convergence of the mean, and divergence of sigma
        training_steps = 50000
    else:
        video_path = os.path.join(results_dir, "kl_loss_fitting-span_both.mp4")
        data = torch.tensor(np.linspace(-0.2, 8.2, N))  # Spanning both
        lr = 1e-3
        training_steps = 300  # Fast convergence
elif args["type"] == "grad_estimator":
    video_path = os.path.join(results_dir, "kl_loss_fitting-grad_estimator.mp4")
    lr = 1e-3
    training_steps = 300  # Fast convergence
else:
    raise Exception('Arg type: {} unknown'.format(args["type"]))
writer.setup(fig, video_path)

neptune.init('morgangiraud/sandbox')
with neptune.create_experiment(
    name='kl_loss',
    description='Experiment to check KL minimization',
    params={
        'span': args['span'], 'type': args['type'], 'lr': lr
    }
):

    # Training
    print('mu {} -> {}'.format(w_mu, mu))
    print('sigma {} -> {}'.format(w_sigma, sigma))
    print("using type: {}".format(args["type"]))
    optimizer = torch.optim.SGD([w_mu, w_sigma], lr=lr, momentum=0.9)
    for i in range(training_steps):
        optimizer.zero_grad()

        if args["type"] == "rsum":
            loss = kl_loss_rsum(dist_q, dist_p, data, delta)
        else:
            loss = kl_loss_grad_estimator(dist_q, dist_p)
        loss.backward()
        optimizer.step()

        neptune.send_metric('loss', loss.item())
        neptune.send_metric('w_mu', w_mu.item())
        neptune.send_metric('w_sigma', w_sigma.item())

        if i % 10 == 0:
            print("loss: {}, w_mu: {}, w_sigma:{}".format(loss.item(), w_mu.item(), w_sigma.item()))

            plt.clf()
            sns.distplot(dist_p.sample([N]).numpy(), label="dist_p")
            ax = sns.distplot(dist_q.sample([N]).numpy(), label="dist_q")
            ax.set_ylim(0, 2)
            ax.set_xlim(-2, 10)
            plt.legend()

            writer.grab_frame()
    writer.finish()
    neptune.send_artifact(video_path)
