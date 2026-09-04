"""Shared Matplotlib style: Times New Roman, print-legible, no clashes."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

FONTDIR = os.path.join(os.path.dirname(__file__), '..', 'fonts')
for f in ['times.ttf', 'timesbd.ttf', 'timesi.ttf']:
    p = os.path.join(FONTDIR, f)
    if os.path.exists(p):
        fm.fontManager.addfont(p)

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.frameon': False,
    'figure.dpi': 200,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# colour-blind-safe qualitative palette
C = {
    'blue': '#1f77b4', 'orange': '#e8862a', 'green': '#2ca02c',
    'red': '#d62728', 'purple': '#7b5cb0', 'teal': '#17a2b8',
    'gray': '#666666', 'gold': '#c9a227', 'navy': '#12355b',
}
SEQ = ['#0b3d5c', '#1f6f8b', '#2a9d8f', '#7cb518', '#e8b400',
       '#e8862a', '#d1495b']


def panel_label(ax, s, x=-0.20, y=1.03, **kw):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='bottom', ha='left', **kw)
