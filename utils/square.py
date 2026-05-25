import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc

if __name__ == '__main__':
    ### data generation
    n_bc = 4
    n_data_per_bc = 25
    #
    engine = qmc.LatinHypercube(d=1)
    data = np.zeros([4, 25, 3])
    for i, j in zip(range(n_bc), [-1, +1, -1, +1]):
        points = (engine.random(n=n_data_per_bc)[:, 0] - 0.5) * 2
        if i < 2:
            data[i, :, 0] = j
            data[i, :, 1] = points
        else:
            data[i, :, 0] = points
            data[i, :, 1] = j
    # BC Values
    # normalized in [0, 1]
    data[0, :, 2] = 1.
    data[2, :, 2] = 50/75
    data = data.reshape(n_data_per_bc * n_bc, 3)
    #
    boundary_data = pd.DataFrame(data).rename(columns={0: "x", 1: "y", 2: "t"})
    #
    Nc = 1000
    engine = qmc.LatinHypercube(d=2)
    colloc = engine.random(n=Nc)
    colloc = 2 * (colloc -0.5)
    #
    domain_data = pd.DataFrame(colloc).rename(columns={0: "x", 1: "y"})
    #
    plt.title("Boundary Data points and Collocation points")
    plt.scatter(data[:, 0], data[:, 1], marker="x", c="k", label="BDP")
    plt.scatter(colloc[:, 0], colloc[:, 1], s=.2, marker=".", c="r", label="CP")
    plt.show()