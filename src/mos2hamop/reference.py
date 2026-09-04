"""Physical reference blocks for residual learning.

Each frame-aligned orbital block is written as a smooth reference plus a
small environment-dependent residual, and the MLP learns only the
residual. The reference is the average block over training pairs of the
same type binned by interatomic distance (a single bin for on-site
blocks). This removes the large, nearly geometry-independent part of the
block (in particular the deep Mo 4s/4p and S 3s semicore levels on the
diagonal), so the residual has a small dynamic range and is learned
accurately from a modest dataset.
"""
import numpy as np


class DistanceReference:
    def __init__(self, dmin, dmax, nbin, ni, nj, onsite=False):
        self.onsite = onsite
        self.ni, self.nj = ni, nj
        if onsite:
            self.edges = np.array([-1.0, 1.0])
            self.nbin = 1
        else:
            self.edges = np.linspace(dmin, dmax, nbin + 1)
            self.nbin = nbin
        self.ref = np.zeros((self.nbin, ni * nj))
        self.count = np.zeros(self.nbin)

    def _bin(self, dist):
        if self.onsite:
            return np.zeros(len(dist), int)
        b = np.digitize(dist, self.edges) - 1
        return np.clip(b, 0, self.nbin - 1)

    def fit(self, dist, Y):
        b = self._bin(dist)
        for k in range(self.nbin):
            sel = b == k
            if sel.sum() > 0:
                self.ref[k] = Y[sel].mean(0)
                self.count[k] = sel.sum()
        # fill empty bins from nearest populated bin
        pop = np.nonzero(self.count > 0)[0]
        for k in range(self.nbin):
            if self.count[k] == 0 and len(pop):
                self.ref[k] = self.ref[pop[np.argmin(abs(pop - k))]]

    def value(self, dist):
        return self.ref[self._bin(np.atleast_1d(dist))]

    def state(self):
        return dict(edges=self.edges, ref=self.ref, count=self.count,
                    onsite=self.onsite, ni=self.ni, nj=self.nj, nbin=self.nbin)

    @classmethod
    def load(cls, st):
        o = cls(0, 1, 1, st['ni'], st['nj'], st['onsite'])
        o.edges = st['edges']; o.ref = st['ref']; o.count = st['count']
        o.nbin = st['nbin']
        return o
