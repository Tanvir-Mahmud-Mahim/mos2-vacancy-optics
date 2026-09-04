"""Ablation and state-of-the-art comparison for the learned Hamiltonian.

All numbers are genuine held-out RMSE on the same 85/15 by-structure split
(scripts/ml_ablation.py -> data/ablation.json).

(a) Held-out RMSE built up from the conventional two-center tight-binding
    baseline to the proposed model.
(b) The decisive contrast: onsite (defect-bearing) vs pair (hopping) blocks,
    conventional two-center vs proposed.
(c) Block-resolved improvement factor of the proposed model over the
    conventional two-center baseline.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import figstyle as F
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
ONSITE = ['onsite_42_42', 'onsite_16_16']
PAIR = ['pair_42_42', 'pair_42_16', 'pair_16_16']
# proposed hybrid: environment residual on onsite, two-center residual on pairs
RULE = {'onsite_42_42': 'E_ref_envMLP', 'onsite_16_16': 'E_ref_envMLP',
        'pair_42_42': 'D_ref_scalarMLP', 'pair_42_16': 'D_ref_scalarMLP',
        'pair_16_16': 'D_ref_scalarMLP'}
PRETTY = {'onsite_42_42': 'Mo onsite', 'onsite_16_16': 'S onsite',
          'pair_42_42': 'Mo–Mo', 'pair_42_16': 'Mo–S',
          'pair_16_16': 'S–S'}


def wrmse(pt, variant_of, types):
    num = sum(pt[variant_of(t)][t]['n_test'] * pt[variant_of(t)][t]['rmse_meV']**2
              for t in types)
    den = sum(pt[variant_of(t)][t]['n_test'] for t in types)
    return np.sqrt(num / den)


def main():
    d = json.load(open(os.path.join(ROOT, 'ablation.json')))
    pt = d['per_type']
    allt = ONSITE + PAIR

    def V(name):
        return lambda t: name
    prop = lambda t: RULE[t]

    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.5))

    # (a) build-up of overall RMSE
    a = ax[0]
    bars = [('global\nmean', wrmse(pt, V('A_globalmean'), allt), F.C['gray']),
            ('two-center\nTB (SOTA)', wrmse(pt, V('B_distanceTB'), allt), F.C['orange']),
            ('learned\ntwo-center', wrmse(pt, V('D_ref_scalarMLP'), allt), F.C['teal']),
            ('proposed', wrmse(pt, prop, allt), F.C['navy'])]
    labels = [b[0] for b in bars]; vals = [b[1] for b in bars]; cols = [b[2] for b in bars]
    a.bar(range(len(vals)), vals, color=cols, width=0.66, edgecolor='k', linewidth=0.4)
    a.set_xticks(range(len(vals))); a.set_xticklabels(labels, fontsize=6.6)
    a.set_ylabel('held-out RMSE (meV)')
    a.set_ylim(0, max(vals) * 1.18)
    for i, v in enumerate(vals):
        a.text(i, v + max(vals) * 0.02, f'{v:.0f}', ha='center', fontsize=7)
    F.panel_label(a, '(a)')

    # (b) onsite vs pair, SOTA vs proposed
    b = ax[1]
    groups = ['onsite\n(defect states)', 'pair\n(hoppings)']
    sota = [wrmse(pt, V('B_distanceTB'), ONSITE), wrmse(pt, V('B_distanceTB'), PAIR)]
    prop_v = [wrmse(pt, prop, ONSITE), wrmse(pt, prop, PAIR)]
    x = np.arange(2); w = 0.36
    b.bar(x - w/2, sota, w, color=F.C['orange'], edgecolor='k', linewidth=0.4,
          label='two-center TB (SOTA)')
    b.bar(x + w/2, prop_v, w, color=F.C['navy'], edgecolor='k', linewidth=0.4,
          label='proposed')
    for i in range(2):
        b.text(x[i]-w/2, sota[i]+6, f'{sota[i]:.0f}', ha='center', fontsize=6.8)
        b.text(x[i]+w/2, prop_v[i]+6, f'{prop_v[i]:.0f}', ha='center', fontsize=6.8)
    b.set_xticks(x); b.set_xticklabels(groups, fontsize=7)
    b.set_ylabel('held-out RMSE (meV)')
    b.set_ylim(0, max(sota) * 1.22)
    b.legend(fontsize=6.4, loc='upper right', handlelength=1.1)
    # onsite gain label in the clear air above the proposed bar, with a short
    # arrow down to it (no overlap with either bar)
    fac = sota[0] / prop_v[0]
    b.annotate(f'{fac:.0f}×', xy=(x[0] + w/2, prop_v[0] + 18),
               xytext=(x[0] + w/2 + 0.07, sota[0] * 0.52),
               ha='center', fontsize=8.5, color=F.C['navy'], fontweight='bold',
               arrowprops=dict(arrowstyle='->', color=F.C['navy'], lw=0.9))
    F.panel_label(b, '(b)')

    # (c) per-block improvement factor
    c = ax[2]
    facs = []
    for t in allt:
        base = pt['B_distanceTB'][t]['rmse_meV']
        pr = pt[RULE[t]][t]['rmse_meV']
        facs.append(base / pr)
    order = np.argsort(facs)
    ypos = np.arange(len(allt))
    cols2 = [F.C['purple'] if allt[i] in ONSITE else F.C['teal'] for i in order]
    c.barh(ypos, [facs[i] for i in order], color=cols2, edgecolor='k',
           linewidth=0.4, height=0.62)
    c.axvline(1.0, color='0.5', lw=0.8, ls='--')
    c.set_yticks(ypos); c.set_yticklabels([PRETTY[allt[i]] for i in order], fontsize=7)
    c.set_xlabel('improvement over SOTA (×)')
    for i, k in enumerate(order):
        c.text(facs[k] + 0.15, i, f'{facs[k]:.1f}×', va='center', fontsize=6.8)
    c.set_xlim(0, max(facs) * 1.22)
    F.panel_label(c, '(c)')

    fig.tight_layout(w_pad=1.4)
    out = os.path.join(ROOT, '..', 'figures', 'fig_ablation.pdf')
    fig.savefig(out); fig.savefig(out.replace('.pdf', '.png'), dpi=200)
    print('wrote', out)
    print('overall: SOTA %.1f -> proposed %.1f meV (%.2fx)'
          % (wrmse(pt, V('B_distanceTB'), allt), wrmse(pt, prop, allt),
             wrmse(pt, V('B_distanceTB'), allt) / wrmse(pt, prop, allt)))


if __name__ == '__main__':
    main()
