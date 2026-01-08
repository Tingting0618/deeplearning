import numpy as np
import torch

# initialize A (m x n) and compute the mean across columns
# (i.e., for each row) which yields a vector of length m.

if __name__ == "__main__":
	m, n = 4, 3
	A = np.random.RandomState(42).rand(m, n)  # shape (m, n)
	print("A (m x n):")
	print(A)
	print("\nMean across columns (per row) -> shape:", A.mean(axis=1).shape)
	print(A.mean(axis=1))
	print(np.mean(A, 1))

	print("numpy a to torch b:")
	a = np.ones(5)
	b = torch.from_numpy(a)
	np.add(a, 1, out=a)
	print(a)
	print(b)

	print("torch a to numpy b:")
	a2 = torch.ones(5)
	b2 = a2.numpy()
	a2.add_(1)
	print(a2)
	print(b2)

	a = torch.ones(5)
	if torch.cuda.is_available():
		device = torch.device("cuda")          # a CUDA device object
		b = torch.ones_like(a, device=device)  # directly create a tensor on GPU
		a = a.to(device)                       # or just use strings ``.to("cuda")``
		c = a + b
		print(c)
		print(c.to("cpu"))