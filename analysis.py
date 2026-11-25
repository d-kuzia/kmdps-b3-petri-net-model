"""Statistical processing of the experiment (section 3.1.4 of the guide):
transient period (3.1), number of replications (3.5), regression W(H),
analysis of variance for the switching rule and chi-square test of the
law of the waiting time distribution. Requires numpy and scipy."""
from pathlib import Path
from itertools import combinations
from math import ceil
import json
import numpy as np
import pandas as pd
from scipy import stats

DATA = Path(__file__).parent/'data'
x = pd.read_csv(DATA/'replications.csv')
result = {}

# --- number of replications, formula (3.5): p = 20 sigma^2 / eps^2
base = x[x.plan == 'H20']
need = {}
for column, scale, eps in [('p_on', 100, .5), ('mean_w', 1, 1.)]:
    sigma = base[column].std(ddof=1)*scale
    need[column] = dict(sigma=sigma, eps=eps, p=ceil(20*sigma**2/eps**2),
                        r=len(base))
result['replications'] = need

# --- transient period, formula (3.1)
pilot = pd.read_csv(DATA/'pilot.csv')
transient = {}
for column, scale, eps in [('reserve', 100, 1.), ('q', 1, .25), ('w', 1, 2.5)]:
    wide = pilot.pivot(index='t', columns='rep', values=column)*scale
    spread = wide.max(axis=1) - wide.min(axis=1)
    bad = spread[spread >= eps]
    T = float(bad.index.max()) + 500 if len(bad) else float(spread.index.min())
    transient[column] = dict(eps=eps, T=T, final_spread=float(spread.iloc[-1]),
                             at20=float(spread.loc[20000]), last=float(spread.index.max()))
result['transient'] = transient

# --- regression W(H) for 10 <= H <= 24
h = x[x.plan.str.match(r'^H\d+$')].copy()
h['H'] = h.plan.str[1:].astype(int)
levels = h[h.H <= 24].groupby('H').mean_w.mean()
H, W = levels.index.values.astype(float), levels.values
Xm = np.column_stack([np.ones_like(H), H])
b = np.linalg.solve(Xm.T @ Xm, Xm.T @ W)                 # formula (3.9)
fit = Xm @ b
r2 = 1 - ((W-fit)**2).sum()/((W-W.mean())**2).sum()
lev25 = h.groupby('H').mean_w.mean()
result['regression'] = dict(b0=b[0], b1=b[1], R2=r2, R=np.sqrt(r2),
                            levels={int(k): v for k, v in levels.items()},
                            resid_max=float(np.abs(W-fit).max()),
                            pred25=float(b[0]+25*b[1]), obs25=float(lev25[25]),
                            pred26=float(b[0]+26*b[1]), obs26=float(lev25[26]))

# --- one-factor ANOVA: switching rule delta = 0 or 5 (D_F on q-1, D_R on q(p-1) degrees of freedom)
anova = {}
for column in ('switches_s', 'p_on', 'mean_w'):
    groups = [x[x.plan == p][column].values for p in ('H20', 'band5')]
    q, p = len(groups), len(groups[0])
    grand = np.mean(np.concatenate(groups))
    DF = p*sum((g.mean()-grand)**2 for g in groups)/(q-1)
    DR = sum(((g-g.mean())**2).sum() for g in groups)/(q*(p-1))
    F = DF/DR
    anova[column] = dict(means=[g.mean() for g in groups], DF=DF, DR=DR, F=F,
                         Fcrit=stats.f.ppf(.95, q-1, q*(p-1)),
                         df=(q-1, q*(p-1)), pvalue=stats.f.sf(F, q-1, q*(p-1)))
result['anova'] = anova

# --- the two rules use the same seeds in every replication, so the samples
# are paired: t-test on 25 differences (equivalent to ANOVA with blocks)
paired = {}
for column in ('switches_s', 'p_on', 'mean_w'):
    a0 = x[x.plan == 'H20'].set_index('rep')[column]
    a5 = x[x.plan == 'band5'].set_index('rep')[column]
    d = (a0 - a5).values
    t = d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))
    paired[column] = dict(mean_diff=d.mean(), sd_diff=d.std(ddof=1), t=t,
                          df=len(d)-1, tcrit=stats.t.ppf(.975, len(d)-1),
                          pvalue=2*stats.t.sf(abs(t), len(d)-1))
result['paired'] = paired

# --- warm-up check for the reserve share: 20 s versus 113 s (same window)
warm = DATA/'warmup-check.csv'
if warm.exists():
    wc = pd.read_csv(warm)
    d = (wc.p_on_113 - wc.p_on_20).values*100
    result['warmup'] = dict(mean20=wc.p_on_20.mean()*100,
                            mean113=wc.p_on_113.mean()*100,
                            diff=d.mean(),
                            half=stats.t.ppf(.975, len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)),
                            r=len(d))

# --- queue length in B: time-weighted mean, deviation (formula (3.3)),
# smallest and largest length met in the observation windows of the base runs
dist = pd.read_csv(DATA/'distribution.csv')
share = dist.groupby('n').p.mean()
mq = (share*share.index).sum()
met = share[share > 0].index
result['queue'] = dict(mean=mq, sd=float(np.sqrt((share*(share.index - mq)**2).sum())),
                       min=int(met.min()), max=int(met.max()))

# --- queues of the two routes: is the difference systematic and what causes it
routes = {}
for plan in ('H20', 'confirm20'):
    g = x[x.plan == plan]
    routes[plan] = dict(q1=g.q1.mean(), q2=g.q2.mean(), longer1=int((g.q1 > g.q2).sum()))
rc = DATA/'route-check.csv'
if rc.exists():
    g = pd.read_csv(rc)
    routes['bc1_constant'] = dict(q1=g.q1.mean(), q2=g.q2.mean(),
                                  longer1=int((g.q1 > g.q2).sum()))
result['routes'] = routes

# --- chi-square test of the waiting time law
s = pd.read_csv(DATA/'wait-sample.csv')
w = s.wait.values
n = len(w)
lag1 = []
for _, g in s.groupby('rep'):
    v = g.wait.values
    lag1.append(np.corrcoef(v[:-1], v[1:])[0, 1])
k = 12
N_SIM = 100000


def chi2_test(v, law):
    """Chi-square goodness of fit with the parameters estimated from v:
    uniform law on [min; max] (every observation must lie inside the support)
    or normal law with the sample mean and deviation; k equal intervals on
    [0; max], sparse classes merged so that every expected count is >= 5."""
    m = len(v)
    lo, hi = v.min(), v.max()
    if law == 'uniform':
        par = dict(a=lo, b=hi)
        cdf = lambda t: np.clip((t - lo)/(hi - lo), 0, 1)
    else:
        par = dict(mu=v.mean(), sd=v.std(ddof=1))
        cdf = lambda t: stats.norm.cdf(t, par['mu'], par['sd'])
    edges = np.linspace(0, hi + 1e-9, k+1)
    observed, _ = np.histogram(v, edges)
    c = cdf(edges)
    probs = np.diff(c)
    probs[0] += c[0]
    probs[-1] += 1 - c[-1]
    expected = m*probs
    obs, exp = [], []
    o_acc = e_acc = 0
    for o, e in zip(observed, expected):
        o_acc += o
        e_acc += e
        if e_acc >= 5:
            obs.append(o_acc)
            exp.append(e_acc)
            o_acc = e_acc = 0
    if e_acc:
        obs[-1] += o_acc
        exp[-1] += e_acc
    obs, exp = np.array(obs), np.array(exp)
    stat = ((obs-exp)**2/exp).sum()
    df = len(obs) - 1 - 2
    return dict(par, chi2=stat, df=df, crit=stats.chi2.ppf(.95, df),
                pvalue=stats.chi2.sf(stat, df), classes=len(obs),
                edges=edges.tolist(), observed=observed.tolist(),
                expected=expected.tolist())


lo, hi = w.min(), w.max()
sd = w.std(ddof=1)
tests = {law: chi2_test(w, law) for law in ('uniform', 'normal')}
# the bounds and the intervals come from the same sample, so the table
# distribution with df = classes - 1 - 2 is only an approximation; the
# p-value of the uniform hypothesis is checked by simulation: N_SIM uniform
# samples of the same size, each passed through exactly the same procedure
# (the normal hypothesis is rejected by a factor of eight, no check needed)
rng = np.random.default_rng(20261125)
for law, t in tests.items():
    if law != 'uniform':
        continue
    sim = np.array([chi2_test(rng.uniform(t['a'], t['b'], n), law)['chi2']
                    for _ in range(N_SIM)])
    t.update(n_sim=N_SIM, exceed=int((sim >= t['chi2']).sum()),
             p_mc=((sim >= t['chi2']).sum() + 1)/(N_SIM + 1),
             crit_mc=float(np.quantile(sim, .95)), sim_mean=float(sim.mean()))
# the same hypotheses separately for each route
by_route = {}
for route, g in s.groupby('route'):
    v = g.wait.values
    r = {}
    for name, dist in (('uniform', stats.uniform(v.min(), v.max()-v.min())),
                       ('normal', stats.norm(v.mean(), v.std(ddof=1)))):
        e10 = dist.ppf(np.linspace(0, 1, 11))
        e10[0], e10[-1] = -np.inf, np.inf
        o10, _ = np.histogram(v, e10)
        stat = ((o10 - len(v)/10)**2/(len(v)/10)).sum()
        r[name] = dict(chi2=stat, df=7, pvalue=stats.chi2.sf(stat, 7))
    by_route[int(route)] = dict(n=len(v), mean=v.mean(), sd=v.std(ddof=1), **r)
result['chi2'] = dict(n=n, zeros=int((w == 0).sum()), by_route=by_route, thin=int(s.k.diff().median()), mean=w.mean(), sd=sd,
                      min=lo, max=hi, lag1_mean=float(np.mean(lag1)),
                      lag1_max=float(np.max(np.abs(lag1))), tests=tests)

# --- point estimates with Student intervals for the base regime
t975 = stats.t.ppf(.975, len(base)-1)
est = {}
for column in ('p_on', 'mean_q', 'mean_w', 'rate', 'switches_s', 'fast_share', 'q1', 'q2'):
    v = base[column]
    est[column] = dict(mean=v.mean(), half=t975*v.std(ddof=1)/np.sqrt(len(v)))
c20 = x[x.plan == 'confirm20'].set_index('rep')
c15 = x[x.plan == 'confirm15'].set_index('rep')
d = c20.mean_w - c15.mean_w
est['paired_saving'] = dict(mean=d.mean(), half=t975*d.std(ddof=1)/np.sqrt(len(d)),
                            min=d.min())
result['estimates'] = dict(t975=t975, **est)

with (DATA/'analysis.json').open('w', encoding='utf-8') as f:
    json.dump(result, f, indent=1, default=float)
print(json.dumps({k: v for k, v in result.items() if k != 'chi2'}, indent=1, default=float))
print(json.dumps({k: v for k, v in result['chi2'].items() if k != 'tests'}, default=float))
for name, t in result['chi2']['tests'].items():
    print(name, {k: v for k, v in t.items() if k not in ('edges', 'observed', 'expected')})
