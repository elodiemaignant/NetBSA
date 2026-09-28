import numpy as np
n_samples, n_nodes = 16, 6

F = lambda t : (t > 0.) * t

network = lambda s, t : np.array([[0.   , 1.   , F(t) , 1.   , F(t) , 1.   , F(t) ],
                                  [1.   , 0.   , s    , F(-t), 0.   , F(-t), s    ],
                                  [F(t) , s    , 0.   , s    , 0.   , 0.   , 0.   ],
                                  [1.   , F(-t), s    , 0.   , s    , F(-t), 0.   ],
                                  [F(t) , 0.   , 0.   , s    , 0.   , s    , 0.   ],
                                  [1.   , F(-t), 0.   , F(-t), s    , 0.   , s    ],
                                  [F(t) , s    , 0.   , 0.   , 0.   , s    , 0.   ]])

network = lambda s, t : np.array([[0.   , F(t) , 1.   , 0.   , 0.   , F(-t)],
                                  [F(t) , 0.   , 1.   , 0.   , F(-t), 0.   ],
                                  [1.   , 1.   , 0.   , s    , 0.   , 0.   ],
                                  [0.   , 0.   , s    , 0.   , 1.   , 1.   ],
                                  [0.   , F(-t), 0.   , 1.   , 0.   , F(t) ],
                                  [F(-t), 0.   , 0.   , 1.   , F(t) , 0.   ]])

S = np.sort(np.random.normal(1., .3, n_samples))
T = np.random.normal(0., .3, n_samples)

networks = [network(s, t) for s, t in zip(S, T)]


with open('Datasets/bsa_vs_pca.npy', 'wb') as file: np.save(file, networks)
