"""Generate supplementary LaTeX tables from the analysis outputs."""
import os, sys, json
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
PAP = os.path.join(os.path.dirname(__file__), '..', 'paper')


def ml_table():
    rep = json.load(open(os.path.join(ROOT, 'ml_report.json')))
    names = {'onsite_42_42': 'Mo on-site', 'onsite_16_16': 'S on-site',
             'pair_42_42': 'Mo--Mo pair', 'pair_42_16': 'Mo--S pair',
             'pair_16_16': 'S--S pair'}
    rows = []
    for t in ['onsite_42_42', 'onsite_16_16', 'pair_42_42', 'pair_42_16',
              'pair_16_16']:
        if t in rep:
            rows.append(f"{names[t]} & {rep[t]['n_test']} & "
                        f"{rep[t]['rmse_meV']:.1f} \\\\")
    body = "\n".join(rows)
    tex = ("\\begin{tabular}{lrr}\n\\toprule\n"
           "Block type & Held-out blocks & RMSE (meV) \\\\\n\\midrule\n"
           f"{body}\n\\bottomrule\n\\end{{tabular}}\n")
    open(os.path.join(PAP, 'tab_ml.tex'), 'w').write(tex)


def config_table():
    an = json.load(open(os.path.join(ROOT, 'dft_analysis.json')))
    res = sorted(an['results'], key=lambda r: (r['nvac'], r['name']))
    rows = []
    for r in res:
        nm = r['name'].replace('_', '\\_')
        rows.append(f"{nm} & {r['nvac']} & "
                    f"{r['a_sub']:.1f} & {r['sig_dc']:.3f} & "
                    f"{r['n_ingap']:.2f} \\\\")
    body = "\n".join(rows)
    tex = ("\\small\n\\begin{tabular}{lrrrr}\n\\toprule\n"
           "Configuration & $n_{\\mathrm{vac}}$ & $A_{\\mathrm{sub}}$ & "
           "$\\sigma(\\omega\\!\\to\\!0)$ & mid-gap/$k$ \\\\\n\\midrule\n"
           f"{body}\n\\bottomrule\n\\end{{tabular}}\n")
    open(os.path.join(PAP, 'tab_configs.tex'), 'w').write(tex)


if __name__ == '__main__':
    ml_table(); config_table()
    print('tables written')
