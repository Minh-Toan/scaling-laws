import torch
import numpy as np
import sgd
import os


'''
sgd training, student width = kc
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



eta = 0.003 
n_epoch = 10000
n_batch = 3

# a sequence of (p+1) alphas, from a to b 
a, b, p = 2, 70, 20
r = (b/a)**(1/p) 
alphas = np.array([a*r**i for i in range(p+1)])
alphas = alphas[:12]
widths = [31, 35, 39, 43, 47, 52, 57, 63, 69, 76, 84, 92] # kc given by theory

folder = f'sgd_data_beta_{beta}_kc_width'
os.makedirs(folder, exist_ok=True)

for iter in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    W0 = torch.from_numpy(ortho_mat(k, d).T).float().to(device)
    n_test = 10000
    X_test  = torch.randn(n_test, d, device=device)
    Y_test  = sig(X_test @ W0) @ v0 

    Es_test = []
    kcs_sgd = []
    for alpha, width in zip(alphas, widths):
        n = int(alpha*d**2)  
        X_train = torch.randn(n, d, device=device)
        Y_train = sig(X_train @ W0) @ v0 + torch.randn(n, device=device)*Delta**(.5)
            
        v_init = torch.full((k,), 0.0, device=device)

        v_init = torch.full((width,), width**-.5, device=device)
        E_train, E_test, Wf, vf = sgd.train2_fast(d, v_init[:width], n, sig, n_batch, n_epoch, eta, X_train, Y_train, X_test, Y_test, normalize=True, device=device)
        Es_test.append(E_test[-1])

        kc_sgd = 0 # number of learnable features by sgd
        W0_ = np.array(W0.cpu()).T
        Q = W0_@np.array(Wf).T
        for i in range(k):
            if np.max(Q[i]) > 0.7:
                kc_sgd += 1
        kcs_sgd.append(kc_sgd)
        
    np.savetxt(f'{folder}/sgd_err_{iter}.csv', Es_test, delimiter=',')
    np.savetxt(f'{folder}/sgd_kc_{iter}.csv', kcs_sgd, fmt='%d', delimiter=',')


