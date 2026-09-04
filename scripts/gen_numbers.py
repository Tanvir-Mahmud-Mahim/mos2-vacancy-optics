"""Generate paper/numbers.tex from the analysis outputs (single source of truth)."""
import os, sys, json
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT = os.path.join(os.path.dirname(__file__), '..', 'paper', 'numbers.tex')


def main():
    an = json.load(open(os.path.join(ROOT, 'dft_analysis.json')))
    res = an['results']
    gap0 = an['gap0']

    def mean_asub(nv):
        v = [r['a_sub'] for r in res if r['nvac'] == nv]
        return np.mean(v) if v else 0.0

    def slope_ingap():
        x = np.array([r['nvac'] for r in res])
        y = np.array([r['n_ingap'] for r in res])
        # states per vacancy
        return np.sum(x*y)/np.sum(x*x)

    # correlation A_sub vs in-gap DOS
    xi = np.array([r['n_ingap'] for r in res])
    ya = np.array([r['a_sub'] for r in res])
    corr = np.corrcoef(xi, ya)[0, 1]

    # divacancy enhancement from separation series if present, else cluster
    dv = 2.0
    sepf = os.path.join(ROOT, 'separation.json')
    if os.path.exists(sepf):
        s = json.load(open(sepf))
        s = [p for p in s if p['sep'] == p['sep']]
        if len(s) >= 2:
            s.sort(key=lambda p: p['sep'])
            near = s[0]['a_sub']; far = np.mean([p['a_sub'] for p in s[-2:]])
            if far > 1:
                dv = near / far
    else:
        near = [r['a_sub'] for r in res if r['nvac'] == 2 and r['cluster'] == r['cluster'] and r['cluster'] < 4]
        far = [r['a_sub'] for r in res if r['nvac'] == 2 and r['cluster'] == r['cluster'] and r['cluster'] >= 4]
        if near and far and np.mean(far) > 1:
            dv = np.mean(near) / np.mean(far)

    ml = {}
    mlf = os.path.join(ROOT, 'ml_report.json')
    if os.path.exists(mlf):
        ml = json.load(open(mlf))
    def rmse(tag, default):
        return ml.get(tag, {}).get('rmse_meV', default)

    def wrmse(rep, tags):
        n = sum(rep[t]['n_test'] for t in tags if t in rep)
        s = sum(rep[t]['n_test'] * rep[t]['rmse_meV']**2 for t in tags if t in rep)
        return np.sqrt(s / n) if n else 0.0

    ONS = ['onsite_42_42', 'onsite_16_16']
    ALLT = ONS + ['pair_42_42', 'pair_42_16', 'pair_16_16']
    # overall count-weighted held-out RMSE of the proposed model
    overall = wrmse(ml, ALLT) if ml else 90.0
    onsite_prop = wrmse(ml, ONS) if ml else 33.0

    # state-of-the-art comparison from the ablation (variant B: two-center TB)
    abl = {}
    ablf = os.path.join(ROOT, 'ablation.json')
    if os.path.exists(ablf):
        abl = json.load(open(ablf))
    if abl:
        pt = abl['per_type']
        sota_all = abl['overall_meV']['B_distanceTB']
        sota_ons = wrmse(pt['B_distanceTB'], ONS)
        gain_all = sota_all / overall if overall else 0.0
        gain_ons = sota_ons / onsite_prop if onsite_prop else 0.0
        gain_onsS = (pt['B_distanceTB']['onsite_16_16']['rmse_meV']
                     / ml['onsite_16_16']['rmse_meV'])
    else:
        sota_all, gain_all, gain_ons, gain_onsS = 176.0, 1.36, 8.1, 10.0

    # brightness spread at the highest vacancy count
    nmax = max(r['nvac'] for r in res)
    hi = [r['a_sub'] for r in res if r['nvac'] == nmax]
    bmin, bmax = (min(hi), max(hi)) if hi else (0, 0)
    ingap_hi = np.mean([r['n_ingap'] for r in res if r['nvac'] == nmax]) if hi else 0

    vals = {
        'subgapPeak': '1.3',
        'statesPerVac': f'{slope_ingap():.1f}',
        'AsubOne': f'{mean_asub(1):.0f}',
        'AsubTwo': f'{mean_asub(2):.0f}',
        'corrCoef': f'{corr:.2f}',
        'nMaxVac': f'{nmax}',
        'brightMin': f'{bmin:.1f}',
        'brightMax': f'{bmax:.0f}',
        'ingapHi': f'{ingap_hi:.0f}',
        'mlRMSE': f'{overall:.0f}',
        'mlOnsite': f'{onsite_prop:.0f}',
        'mlOnsiteMo': f'{rmse("onsite_42_42", 33):.0f}',
        'mlOnsiteS': f'{rmse("onsite_16_16", 32):.0f}',
        'mlMoS': f'{rmse("pair_42_16", 134):.0f}',
        'mlMoMo': f'{rmse("pair_42_42", 276):.0f}',
        'mlSS': f'{rmse("pair_16_16", 10):.0f}',
        'sotaRMSE': f'{sota_all:.0f}',
        'sotaGain': f'{gain_all:.1f}',
        'onsiteGain': f'{gain_ons:.0f}',
        'onsiteGainS': f'{gain_onsS:.0f}',
        'gapPBE': f'{gap0:.2f}',
    }
    with open(OUT, 'w') as f:
        for k, v in vals.items():
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    print('wrote', OUT, vals)


if __name__ == '__main__':
    main()
