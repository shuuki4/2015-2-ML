import numpy as np

result = np.ones((3,4), dtype=int)
for j in range(3) :
	vote = np.bincount(result[j,:]).argmax()
	print vote