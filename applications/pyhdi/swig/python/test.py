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

    pyhdi.set_n_points(a.shape[0])
    pyhdi.set_input(a)
    pyhdi.set_output(b)

    # perplexity = 30.0
    # seed = 0
    # minimum_gain = 0.1
    # eta = 200.0
    # momentum = 0.5
    # final_momentum = 0.8
    # mom_switching_iter = 250
    # exaggeration_factor = 4
    # remove_exaggeration_iter = 250
    # iterations = 500
    # pyhdi.run_tsne(perplexity, seed, minimum_gain, eta, momentum, final_momentum, mom_switching_iter, exaggeration_factor, remove_exaggeration_iter, iterations)

    perplexity = 30.0
    theta = 0.5
    exaggeration_iter = 450
    iterations = 1000
    pyhdi.run_asne(perplexity, theta, exaggeration_iter, iterations)

    import matplotlib.pyplot as plt
    plt.scatter(np.array(b[:, 0]), np.array(b[:, 1]), c=y)
    plt.savefig('asdf.png')
except pyhdi.PyHDIException as e:
    print(type(e), e)