"""Per-block-type MLP regression of Hamiltonian and overlap blocks.

Five block types: onsite-Mo, onsite-S, and the ordered pairs Mo-Mo,
Mo-S, S-S (S-Mo blocks follow from Hermiticity of the assembled
matrix). Each model maps the 340-dimensional pair-frame descriptor to
the flattened orbital block, rotated into the pair frame. Targets are
standardized element-wise; training uses Adam with early stopping on a
validation split. Everything runs on CPU.
"""
import numpy as np
import torch
import torch.nn as nn

from .blocks import NAO
from .rotations import atom_rotation

BLOCK_TYPES = [('onsite', 42, 42), ('onsite', 16, 16),
               ('pair', 42, 42), ('pair', 42, 16), ('pair', 16, 16)]

# Physics-informed descriptor selection. The block ablation (ml_ablation.py)
# shows that the onsite blocks, which carry the environment-dependent defect
# potential, need the full equivariant environment descriptor, whereas the
# pair (hopping) blocks are governed by the two-center geometry and are
# reproduced from the frame-invariant displacement scalars alone. Using each
# ingredient where the physics demands it gives the lowest held-out error and
# avoids overfitting the many pair blocks with the high-dimensional cloud.
SCALAR_COLS = 4          # [dist, dx, dz, dz*dz]: the two-center displacement


def select_features(kind, X):
    """Full environment descriptor for onsite blocks; two-center scalars for
    pair (hopping) blocks."""
    return X if kind == 'onsite' else X[:, :SCALAR_COLS]


def type_key(kind, Zi, Zj):
    if kind == 'onsite':
        return ('onsite', Zi, Zi)
    if (Zi, Zj) == (16, 42):
        return None  # obtained from the transpose of the (42, 16) pair
    return ('pair', Zi, Zj)


def rotate_into_frame(block, Zi, Zj, theta):
    """Rotate a global-frame block into the pair frame (rotation by -theta)."""
    Di = atom_rotation(Zi, -theta)
    Dj = atom_rotation(Zj, -theta)
    return Di @ block @ Dj.T


def rotate_out_of_frame(block, Zi, Zj, theta):
    Di = atom_rotation(Zi, theta)
    Dj = atom_rotation(Zj, theta)
    return Di @ block @ Dj.T


class BlockMLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=(320, 320)):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BlockModel:
    """One standardized MLP for one block type and one target (H or S)."""

    def __init__(self, n_in, ni, nj):
        self.ni, self.nj = ni, nj
        self.net = BlockMLP(n_in, ni * nj)
        self.x_mean = self.x_std = self.y_mean = self.y_std = None

    def fit(self, X, Y, epochs=400, lr=1e-3, batch=512, seed=0, log=print,
            val_frac=0.1, patience=40):
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        n = len(X)
        idx = rng.permutation(n)
        nval = max(1, int(val_frac * n))
        vi, ti = idx[:nval], idx[nval:]
        self.x_mean = X[ti].mean(0); self.x_std = X[ti].std(0) + 1e-8
        self.y_mean = Y[ti].mean(0); self.y_std = Y[ti].std(0) + 1e-6
        Xn = (X - self.x_mean) / self.x_std
        Yn = (Y - self.y_mean) / self.y_std
        Xt = torch.tensor(Xn[ti], dtype=torch.float32)
        Yt = torch.tensor(Yn[ti], dtype=torch.float32)
        Xv = torch.tensor(Xn[vi], dtype=torch.float32)
        Yv = torch.tensor(Yn[vi], dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5,
                                                           patience=10)
        best, best_state, bad = np.inf, None, 0
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(len(Xt))
            for b in range(0, len(Xt), batch):
                sel = perm[b:b + batch]
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.net(Xt[sel]), Yt[sel])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vloss = nn.functional.mse_loss(self.net(Xv), Yv).item()
            sched.step(vloss)
            if vloss < best - 1e-6:
                best, bad = vloss, 0
                best_state = {k: v.clone() for k, v in
                              self.net.state_dict().items()}
            else:
                bad += 1
                if bad > patience:
                    break
            if ep % 25 == 0:
                log(f'  epoch {ep}: val mse (standardized) {vloss:.3e}')
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return best

    def predict(self, X):
        Xn = (X - self.x_mean) / self.x_std
        with torch.no_grad():
            Yn = self.net(torch.tensor(Xn, dtype=torch.float32)).numpy()
        return Yn * self.y_std + self.y_mean

    def state(self):
        return dict(net=self.net.state_dict(), x_mean=self.x_mean,
                    x_std=self.x_std, y_mean=self.y_mean, y_std=self.y_std,
                    ni=self.ni, nj=self.nj)

    def load(self, st):
        self.net.load_state_dict(st['net'])
        self.x_mean, self.x_std = st['x_mean'], st['x_std']
        self.y_mean, self.y_std = st['y_mean'], st['y_std']
