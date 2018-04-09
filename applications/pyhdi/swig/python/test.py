from pyhdi_sklearn import HighDimensionalInspectorTSNE

import numpy as np
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata("MNIST original")
n = 100
X = np.matrix(mnist.data / 255.0)[:n, :]
y = mnist.target[:n]

print(set(y))

print(X.shape)

a = np.asarray(X, dtype=np.double)

for i in range(a.shape[0]):
    a[i, 0] = i

tsne = HighDimensionalInspectorTSNE(method="asne")

b = tsne.fit_transform(a)

import matplotlib.pyplot as plt
plt.scatter(np.array(b[:, 0]), np.array(b[:, 1]), c=y)
plt.savefig('embedding.png')
