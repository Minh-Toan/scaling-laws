import torch
import numpy as np
import sgd
import os

'''
SGD training, student with different widths (k=100)
'''


device = "cuda:1"

def ortho_mat(k, d):
    A = np.random.randn(d, k)
    Q, _ = np.linalg.qr(A)
    return Q.T

d = 200
gamma = 0.5
k = int(gamma*d)

Delta = 0.01
sig = lambda x: torch.tanh(2*x)

beta = 1
v_np = np.array([p**(-beta) for p in range(1, k+1)])
v_np /= np.linalg.norm(v_np)
v0 = torch.tensor(v_np, dtype=torch.float32, device=device)

W0 = torch.from_numpy(ortho_mat(k, d).T).float().to(device)

n_test = 10000
X_test  = torch.randn(n_test, d, device=device)
Y_test  = sig(X_test @ W0) @ v0 

eta = 0.003 
n_epoch = 10000
n_batch = 3

# a sequence of (p+1) alphas, from a to b 
a, b, p = 2, 70, 20
r = (b/a)**(1/p) 
alphas = np.array([a*r**i for i in range(p+1)])

for iter in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    folder = f'sgd_data_beta_{beta}_new/sgd_data_{iter}'
    os.makedirs(folder, exist_ok=True)
    for alpha in alphas:
        n = int(alpha*d**2)  
        X_train = torch.randn(n, d, device=device)
        Y_train = sig(X_train @ W0) @ v0 + torch.randn(n, device=device)*Delta**(.5)
        Es_test = []
        if alpha < 10:
            widths = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        else:
            widths = [100, 90, 80, 70, 60, 50, 40]

        for width in widths:
            v_init = torch.full((width,), 0.0, device=device)
            E_train, E_test, Wf, vf = sgd.train2_fast(d, v_init[:width], n, sig, n_batch, n_epoch, eta, X_train, Y_train, X_test, Y_test, normalize=True, device=device)
            Es_test.append(E_test[-1])        
        np.savetxt(f'{folder}/sgd_err_n_{n}.csv', Es_test, delimiter=',')


