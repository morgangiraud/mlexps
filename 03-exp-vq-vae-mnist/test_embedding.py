"""
A quick file to test embedding pytorch behaviors
"""

import torch
import torch.nn as nn

e = nn.Embedding(5, 2)

print(e.weight)

x = torch.LongTensor([0, 2, 4])
y = e(x)

out = torch.sum(y, axis=1)
loss = (out**2).mean()

sgd = torch.optim.SGD(e.parameters(), 1., 0.)
loss.backward()
sgd.step()

print(e.weight)
print(e.embedding_dim)
print(x.grad)  # None

a = torch.randn(3, 4, 2)

out = torch.cdist(a, e.weight)
print("a.shape: {} x e.shape: {} -> out.shape: {}".format(a.shape, e.weight.shape, out.shape))
out = torch.argmin(out, dim=2)
print(out.shape)
print(out)
out = e(out)
print(out.shape)
