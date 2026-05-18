import torch
import numpy as np


def train(d, v, n, sig, n_batch, n_epoch, eta, X_train, Y_train, X_test, Y_test, normalize, device):
    '''
    training with v fixed
    this is never used in the paper
    '''
    class Student(torch.nn.Module):
        def __init__(self, d, k, v):
            super().__init__()
            self.W = torch.nn.Linear(d, k, bias=False, device=device)
            with torch.no_grad():
                self.W.weight.div_(d**.5)
            self.register_buffer("v", v)

        def forward(self, x):
            h = sig(self.W(x))
            return h @ self.v
    k = len(v)
    batch_size = n // n_batch
    student = Student(d, k, v).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=eta)
    loss_fn = torch.nn.MSELoss()

    errs_train = []
    errs_test = []

    for epoch in range(n_epoch):
        perm = torch.randperm(n, device=device)
        student.train()

        for i in range(n_batch):
            idx = perm[i*batch_size:(i+1)*batch_size]
            loss = loss_fn(student(X_train[idx]), Y_train[idx])

            opt.zero_grad()
            loss.backward()
            opt.step()

        if normalize:
            with torch.no_grad():
                W = student.W.weight
                W.div_(W.norm(dim=1, keepdim=True))

        student.eval()
        with torch.no_grad():
            errs_train.append(loss.item())
            errs_test.append(loss_fn(student(X_test), Y_test).item())

    return np.array(errs_train), np.array(errs_test), student.W.weight.detach().cpu().numpy()


def train2(d, v_init, n, sig, n_batch, n_epoch, eta, X_train, Y_train, X_test, Y_test, normalize, device):
    '''
    SGD training with v learnable
    '''

    class Student(torch.nn.Module):
        def __init__(self, d, k):
            super().__init__()
            self.W = torch.nn.Linear(d, k, bias=False, device=device)
            self.v = torch.nn.Parameter(v_init.clone())

            with torch.no_grad():
                self.W.weight.div_(d**.5)

        def forward(self, x):
            h = sig(self.W(x))
            return h@self.v
    k = len(v_init)
    batch_size = n // n_batch
    student = Student(d, k).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=eta)
    loss_fn = torch.nn.MSELoss()

    errs_train = []
    errs_test = []

    for epoch in range(n_epoch):
        perm = torch.randperm(n, device=device)
        student.train()

        for i in range(n_batch):
            idx = perm[i*batch_size:(i+1)*batch_size]
            loss = loss_fn(student(X_train[idx]), Y_train[idx])

            opt.zero_grad()
            loss.backward()
            opt.step()

        if normalize:
            with torch.no_grad():
                W = student.W.weight
                W.div_(W.norm(dim=1, keepdim=True))

        student.eval()

        with torch.no_grad():
            errs_train.append(loss.item())
            errs_test.append(loss_fn(student(X_test), Y_test).item())

    return np.array(errs_train), np.array(errs_test), student.W.weight.detach().cpu().numpy(), student.v.detach().cpu().numpy()

def train2_fast(d, v_init, n, sig, n_batch, n_epoch, eta, X_train, Y_train, X_test, Y_test, normalize, device):
    '''
    same as train2, slightly faster
    '''

    class Student(torch.nn.Module):
        def __init__(self, d, k):
            super().__init__()
            self.W = torch.nn.Linear(d, k, bias=False, device=device)
            self.v = torch.nn.Parameter(v_init.clone())

            with torch.no_grad():
                self.W.weight.div_(d**0.5)

        def forward(self, x):
            return sig(self.W(x)) @ self.v

    k = len(v_init)
    student = Student(d, k).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=eta)
    loss_fn = torch.nn.MSELoss()

    # Pre-split batches once 
    batch_size = n // n_batch
    batches = [
        slice(i * batch_size, (i + 1) * batch_size)
        for i in range(n_batch)
    ]

    errs_train = []
    errs_test = []

    for epoch in range(n_epoch):
        student.train()

        # No randperm: sequential access (much faster for huge n)
        for sl in batches:
            pred = student(X_train[sl])
            loss = loss_fn(pred, Y_train[sl])

            opt.zero_grad(set_to_none=True)  # slightly faster
            loss.backward()
            opt.step()

        if normalize:
            with torch.no_grad():
                W = student.W.weight
                W.div_(W.norm(dim=1, keepdim=True))

        student.eval()

        with torch.no_grad():
            # last batch loss already computed → reuse
            errs_train.append(loss.item())

            # test is small → fine
            test_pred = student(X_test)
            errs_test.append(loss_fn(test_pred, Y_test).item())

    return (np.array(errs_train),
            np.array(errs_test),
            student.W.weight.detach().cpu().numpy(),
            student.v.detach().cpu().numpy())
