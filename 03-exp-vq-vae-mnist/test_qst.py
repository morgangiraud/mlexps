import torch
from torch.autograd import Variable

from model import VectorQuantStraightThrough

x = Variable(
    torch.tensor([[[[-1.3956, 0.6380, -1.7173], [-1.0479, -1.5089, 0.6566],
                    [0.7305, -0.9510, 0.3985]],
                   [[-0.5415, -0.7786, 0.5485], [1.1408, -0.7515, 1.0064],
                    [-0.5344, 1.2565, 0.6173]]]]),
    requires_grad=True
)
print("x.shape: {}".format(x.shape))

e_w = torch.tensor([
    [0., 0.],  # 0
    [0., 0.5],
    [0., 1.],
    [0.5, 0.],  # 3
    [0.5, 0.5],
    [0.5, 1.],  # 5
    [1., .0],
    [1., .5],
    [1., 1.]
])
e = torch.nn.Embedding(e_w.shape[0], e_w.shape[1], _weight=e_w)
print("e.shape: {}".format(e.weight.shape))
vqst = VectorQuantStraightThrough(7, 2)
# We override the embedding to control the outputs
vqst._embedding = e

x_q, indices = vqst(x)
loss = x_q.mean()

print("x_q.shape: {}".format(x_q.shape))
print("out.shape: {}".format(x_q.shape))

print("\nx[0]:\n", x[0])
print("\nindices[0]:\n", indices.view(-1, 3, 3)[0])
print("\nx_q[0]:\n", x_q[0])

loss.backward()
print(x.grad)
