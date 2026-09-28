import numpy as np
from topologies import complete_graph, random_graph, star_graph, star_2_graph

star_graphs = [np.abs(star_graph(10) + random_graph(n=10, sigma=.05)) for i in range(6)]
star_2_graphs = [np.abs(star_2_graph(10) + random_graph(n=10, sigma=.05)) for i in range(5)]
complete_graphs = [np.abs(.5 * complete_graph(10) + random_graph(n=10, sigma=.05)) for i in range(4)]

dataset = np.array(star_graph + star_2_graphs + complete_graphs)

with open('Datasets/clustered.npy', 'wb') as file: np.save(file, dataset)
