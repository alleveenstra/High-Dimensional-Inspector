import pyhdi

import numpy as np
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata("MNIST original")
n = 100
X = np.matrix(mnist.data / 255.0)[:n, :]
y = mnist.target[:n]

try:
    print(set(y))

    print(X.shape)

    a = np.matrix(X, dtype=np.double)
    b = np.matrix(np.zeros((X.shape[0], 2)), dtype=np.double)
    #b = np.matrix(np.random.randn(X.shape[0], 2), dtype=np.double)

    print(a.shape)
    print(b.shape)

    for i in range(a.shape[0]):
        a[i, 0] = i

    tsne = pyhdi.HDI_tSNE()
    tsne.parameters().set_n_points(a.shape[0])
    tsne.parameters().set_input(a)
    tsne.parameters().set_output(b)
    tsne.run(1000)

    asne = pyhdi.HDI_aSNE()
    asne.parameters().set_n_points(a.shape[0])
    asne.parameters().set_input(a)
    asne.parameters().set_output(b)
    asne.run(1000)

    import matplotlib.pyplot as plt
    plt.scatter(np.array(b[:, 0]), np.array(b[:, 1]), c=y)
    plt.savefig('asdf.png')
except pyhdi.PyHDIException as e:
    print(type(e), e)