"""Write paper/tab_ablation.tex from data/ablation.json (held-out RMSE).

Reports, for each model variant, the count-weighted held-out RMSE overall
and split into onsite (defect-bearing) and pair (hopping) blocks. The
proposed model uses the equivariant environment descriptor on the onsite
blocks and the two-center displacement on the pair blocks.
"""
import os, sys, json
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT = os.path.join(os.path.dirname(__file__), '..', 'paper', 'tab_ablation.tex')
ONS = ['onsite_42_42', 'onsite_16_16']
PAIR = ['pair_42_42', 'pair_42_16', 'pair_16_16']
RULE = {'onsite_42_42': 'E_ref_envMLP', 'onsite_16_16': 'E_ref_envMLP',
        'pair_42_42': 'D_ref_scalarMLP', 'pair_42_16': 'D_ref_scalarMLP',
        'pair_16_16': 'D_ref_scalarMLP'}


def w(pt, vof, tags):
    n = sum(pt[vof(t)][t]['n_test'] for t in tags)
    s = sum(pt[vof(t)][t]['n_test'] * pt[vof(t)][t]['rmse_meV'] ** 2 for t in tags)
    return np.sqrt(s / n)


def main():
    d = json.load(open(os.path.join(ROOT, 'ablation.json')))
    pt = d['per_type']
    const = lambda name: (lambda t: name)
    prop = lambda t: RULE[t]
    rows = [
        ('Global-mean reference', 'none', const('A_globalmean')),
        ('Two-center tight binding (conventional)', 'distance', const('B_distanceTB')),
        ('Environment MLP, no reference', 'environment', const('C_envMLP_noref')),
        ('Two-center learned (reference + displacement)', 'displacement', const('D_ref_scalarMLP')),
        ('Environment learned everywhere', 'environment', const('E_ref_envMLP')),
        (r'\textbf{Proposed (physics-selected)}', 'both', prop),
    ]
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Ablation and comparison against the conventional '
                 r'two-center tight-binding model. Values are count-weighted '
                 r'held-out root-mean-square errors (meV) on an identical '
                 r'85/15 by-structure split, split into the onsite '
                 r'(defect-bearing) and pair (hopping) blocks. The proposed '
                 r'model applies the equivariant environment descriptor to '
                 r'the onsite blocks and the two-center displacement to the '
                 r'pair blocks.}')
    lines.append(r'\label{tab:ablation}')
    lines.append(r'\begin{tabular}{lcccc}')
    lines.append(r'\hline')
    lines.append(r'Model & Descriptor & Overall & Onsite & Pair \\')
    lines.append(r'\hline')
    for name, desc, vof in rows:
        o = w(pt, vof, ONS + PAIR); on = w(pt, vof, ONS); pr = w(pt, vof, PAIR)
        lines.append(f'{name} & {desc} & {o:.0f} & {on:.0f} & {pr:.0f} '
                     r'\\')
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    open(OUT, 'w').write('\n'.join(lines) + '\n')
    print('wrote', OUT)
    for name, desc, vof in rows:
        print(f'{name:46s} overall {w(pt,vof,ONS+PAIR):6.0f}  '
              f'onsite {w(pt,vof,ONS):6.0f}  pair {w(pt,vof,PAIR):6.0f}')


if __name__ == '__main__':
    main()
