# Tutorial url https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py

import torch

# If you set its attribute .requires_grad as True,
# it starts to track all operations on it.
# When you finish your computation you can call .backward()
# and have all the gradients computed automatically
x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z, out)

# Gradients
out.backward()
print(x.grad)  # d(out)/dx

# Vector Jacobian
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)