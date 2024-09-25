
import torch

hid_dim = 32
data = torch.randn(10, 2, 3, hid_dim)
data = data.view(10, 2 * 3, hid_dim)
W = torch.randn(hid_dim)

result = torch.bmm(data.view(-1, hid_dim, 1), W).squeeze(-1).view(10, 2, 3)
