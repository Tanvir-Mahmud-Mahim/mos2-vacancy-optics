"""Validate negf.transmission on a 1D single-orbital chain (analytic T=1)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mos2hamop.negf import transmission

# 1D chain: on-site 0, hopping t=-1, orthogonal (S=I). Band -2..2.
# A clean chain has T(E)=1 for |E|<2. Device = N pristine layers.
t = -1.0
H00 = np.array([[0.0]]); H01 = np.array([[t]])
S00 = np.array([[1.0]]); S01 = np.array([[0.0]])
N = 6
layers_H = [H00.copy() for _ in range(N)]
layers_S = [S00.copy() for _ in range(N)]
coup_H = [H01.copy() for _ in range(N - 1)]
coup_S = [S01.copy() for _ in range(N - 1)]
E = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
T = transmission(E, layers_H, layers_S, coup_H, coup_S, H00, H01, S00, S01,
                 eta=1e-6)
print('E   :', E)
print('T   :', np.round(T, 5))
print('max |T-1| in band:', np.abs(T - 1).max())

# two-orbital chain with a barrier layer to check <1 transmission
H00b = np.diag([0.0, 0.0]); H01b = np.diag([t, t])
S00b = np.eye(2); S01b = np.zeros((2, 2))
lH = [H00b.copy() for _ in range(N)]; lS = [S00b.copy() for _ in range(N)]
lH[3] = np.diag([1.2, 1.2])  # on-site barrier
cH = [H01b.copy() for _ in range(N - 1)]; cS = [S01b.copy() for _ in range(N - 1)]
Tb = transmission(np.array([0.0]), lH, lS, cH, cS, H00b, H01b, S00b, S01b, 1e-6)
print('two-channel clean would be 2; with mid barrier T(0)=', np.round(Tb, 4))
