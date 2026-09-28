import numpy as np 

def random_graph(n, sigma):
    X = np.random.multivariate_normal(mean=np.zeros(n * (n - 1) // 2), cov=sigma * np.eye(n * (n - 1) // 2)).reshape((n, n))
    return(.5 * (X + X.transpose()) - np.diag(np.diagonal(X)))

def complete_graph(n): return(np.ones((n, n)) - np.eye(n))

def star_graph(n):
    X = np.zeros((n, n))
    X[1:, 0] = np.ones(n-1)
    X[0, 1:] = np.ones(n-1)
    return(X)

def cyclic_graph(n): return(np.eye(N=n, k=1) + np.eye(N=n, k=-n+1) + np.eye(N=n, k=-1) + np.eye(N=n, k=n-1))

def wheel_graph(n):
    X = star_graph(n) + np.eye(N=n, k=1) + np.eye(N=n, k=-1)
    X[0, 1] = 1
    X[1, 0] = 1
    X[-1, 1] = 1
    X[1, -1] = 1
    return(X)

def bipartite_graph(k):
    X = complete_graph(2 * k)
    X = np.random.rand(2*k, 2*k)
    X += X.transpose()
    for i in range(2 * k): 
        for j in range(2 * k): 
            if (i - j) % 2 == 0: X[i, j] = 0
    return(X)

def disconnected_graph(n, k=2):
    if n%k>0: return('n cannot be divided in k components')
    X = np.zeros((n, n))
    for c in range(k): 
        B = np.random.randint(0, 2, (n//k, n//k))
        for i in range(n//k): 
            for j in range(i, n//k): B[i, j] = 0
        X[c * (n//k): (c+1) * (n//k), c * (n//k): (c+1) * (n//k)] = B + B.transpose()
    return(X)
    
def star_2_graph(n):
    X = np.zeros((n, n))
    X[1:, 0] = np.ones(n-1)
    X[0, 1:] = np.ones(n-1)
    X[2:, 1] = np.ones(n-2)
    X[1, 2:] = np.ones(n-2)
    return(X)


