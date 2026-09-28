import numpy as np
from topologies import complete_graph

n_nodes = 10

complete = complete_graph(n_nodes)
ref_points = [complete * (X + np.transpose(X)) / 2. for X in np.random.normal(loc=1., scale=.4, size=(3, n_nodes, n_nodes))]
while np.any(np.array(ref_points) < 0.):
    ref_points = [complete * (X + np.transpose(X)) / 2. for X in np.random.normal(loc=1., scale=.4, size=(3, n_nodes, n_nodes))]

with open(f'Datasets/ref_points_n={n_nodes}.npy', 'wb') as f:
    np.save(f, ref_points)
