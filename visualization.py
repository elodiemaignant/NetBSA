import networkx as nx
import numpy as np

def plot_loop(ax, x, y, color, edge_width, zorder):
    theta = np.linspace(0., 2 * np.pi, 100)
    c = x + .2 + .2 * np.cos(theta)
    s = y + .2 * np.sin(theta)
    ax.plot(c, s, lw=edge_width, c=color, zorder=zorder)

def plot_network(ax, X, layout='kamada-kawai', s=150, node_color='r', m='o', edge_width=4.5, x_max=None):
    if x_max == None: x_max = np.max(np.abs(X))
    n_vertices = len(X)
    
    if layout=='circular': 
        theta = np.linspace(0., 2. * np.pi, n_vertices + 1)[:-1]
        vertices = np.transpose(np.vstack((np.cos(theta), np.sin(theta))))
    
    if layout=='kamada-kawai': 
        D = np.sqrt(- np.log(1E-6 * np.ones_like(X) + np.abs(X) / np.max(np.abs(X)) / 10))
        G = nx.from_numpy_array(D, create_using=nx.Graph)
        kk = nx.kamada_kawai_layout(G)
        vertices = np.array([kk[i] for i in range(len(X))])
    
    if layout=='bipartite': 
        l = np.linalg.eigvalsh(X)
        vertices = np.array([[len(X) * (l[i] > 0.) / 2., i] for i in range(len(X))])
    
    ax.scatter(vertices[:, 0], vertices[:, 1], s=s, color=node_color, marker=m, zorder=2)
    
    for i in range(n_vertices):
        if X[i, i] > 0.: 
            alpha=np.abs(X[i, i]) / x_max
            c = (1. - alpha, 1. - alpha, 1. - alpha)
            plot_loop(ax, vertices[i, 0], vertices[i, 1], color=c, edge_width=edge_width, zorder=alpha)
        
        elif X[i, i] < 0.: 
            alpha=np.abs(X[i, i]) / x_max
            c = (1., 1. - alpha, 1. - alpha)
            plot_loop(ax, vertices[i, 0], vertices[i, 1], color=c, edge_width=edge_width, zorder=alpha)
        
        for j in range(i+1, n_vertices):
            if X[i, j] > 0.: 
                alpha=np.abs(X[i, j]) / x_max
                c = (1. - alpha, 1. - alpha, 1. - alpha)
                ax.plot(vertices[[i, j], 0], vertices[[i, j], 1], c=c, lw=edge_width, zorder=alpha)
            
            elif X[i, j] < 0.:
                alpha=np.abs(X[i, j]) / x_max
                c = (1., 1. - alpha, 1. - alpha)
                ax.plot(vertices[[i, j], 0], vertices[[i, j], 1], c=c, lw=edge_width, zorder=alpha)
    
