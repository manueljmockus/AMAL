import torch
from tp1 import mse, linear

# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
test_mse = torch.autograd.gradcheck(mse, (yhat, y), raise_exception=True)
print("test MSE ok? ", test_mse)

X = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
W = torch.randn(5,10, requires_grad=True, dtype=torch.float64)
b = torch.randn(10, requires_grad=True, dtype=torch.float64)
test_linear = torch.autograd.gradcheck(linear, (X, W, b), raise_exception=True)
print("test Linear ok? ", test_linear)

