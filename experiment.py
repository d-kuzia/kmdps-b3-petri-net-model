"""Produce the raw data: verification cases, pilot trajectories and replications."""

from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
import csv
from model import Settings, Relay

OUT = Path(__file__).parent/'data'
BASE = Settings()
REPS = 25          # number of replications, formula (3.5), see analysis.py
PILOTS = 6         # pilot trajectories for formula (3.1)
PLANS = {f'H{h}': replace(BASE, high=h) for h in (10,15,18,20,22,23,24,25,26)}
PLANS['band5'] = replace(BASE, band=5)
PLANS['fast12'] = replace(BASE, fast=12)
PLANS['slow12'] = replace(BASE, arrival=12)
PLANS['dense8'] = replace(BASE, arrival=8)
PLANS['confirm15'] = replace(BASE, high=15)
PLANS['confirm20'] = BASE


def write(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def seed_of(name, j):
    return (73003 if name.startswith('confirm') else 33003) + 97*j


def run_case(job):
    name, j = job
    seed = seed_of(name, j)
    row = Relay(PLANS[name], seed).run()
    hist = row.pop('hist')
    return dict(plan=name, rep=j+1, seed=seed, **row), hist


def run_pilot(j):
    """Cumulative means y_i(t) on [0; t] for one trajectory, formula (3.2)."""
    seed = 17003 + 97*j
    m = Relay(replace(BASE, warm=0), seed)
    grid = [500*k for k in range(1, 441)]
    rows = []

    def record(model):
        t = model.net.t
        while grid and t >= grid[0]:
            g = grid.pop(0)
            rows.append(dict(rep=j+1, seed=seed, t=g,
                             q=sum(model.area_lane)/t, reserve=model.active/t,
                             qa=model.area_a/t,
                             w=model.wait/max(1, model.starts)))
    m.run(record=record)
    return rows


def state(m, t, marks, exits, stamps=None):
    """Put the net into a prepared state: marking and planned exits."""
    for name, value in marks.items():
        m.P[name].mark = value
    m.P['Q'].mark = m.length()          # place Q mirrors the queue in B
    for tr in m.net.transitions:
        tr.exits = list(exits.get(tr.name, []))
    if stamps:
        for i in range(2):
            m.stamps[i].extend(stamps[i])
    m.net.t = t
    m.in_a = (m.length() + m.P['A'].mark + m.busy())
    m.measured = True


def warm_check(j):
    """Reserve share of the base run after discarding 20 s and 113 s
    (formula (3.1) gives 113 s for this response); window 200 s in both."""
    seed = seed_of('H20', j)
    late = replace(BASE, warm=113000, end=313000)
    return dict(rep=j+1, seed=seed, p_on_20=Relay(BASE, seed).run()['p_on'],
                p_on_113=Relay(late, seed).run()['p_on'])


def route_check(j):
    """Base run with a constant BC1 duration (25 ms instead of 25 +/- 3 ms):
    the queues of the two routes, to find the cause of their difference."""
    seed = seed_of('H20', j)
    r = Relay(replace(BASE, jitter_bc=0), seed).run()
    return dict(rep=j+1, seed=seed, q1=r['q1'], q2=r['q2'])


def verify():
    # Simultaneous exits of c1 and c2 at 100 ms with 10 + 10 tokens in B and
    # the reserve on: v1 starts with 15 ms, T_off fires, v2 starts with 25 ms.
    sim = Relay(BASE, 3303)
    state(sim, 100., dict(G=0, B1=10, B2=10, F_B=5, D1=0, D2=0, R0=0, R1=1),
          dict(g=[120.], c1=[100.], c2=[100.]), ([90.]*10, [90.]*10))
    sim.net.step()
    sim.check()
    assert sim.out == 2 and sim.T['c1'].exits == [115.] and sim.T['c2'].exits == [125.]
    assert sim.length() == 18
    assert sim.P['R0'].mark == 1 and sim.fast_starts == 1 and sim.balance() == 0

    # All 25 places of B hold route-1 packets; a route-2 packet with a free
    # BC2 goes straight to the line through p2 and does not need F_B.
    full = Relay(BASE, 3303)
    state(full, 100., dict(G=0, B1=BASE.cap_b, F_B=0, D1=0, E2=0),
          dict(g=[120.], u2=[100.], c1=[110.]), ([90.]*BASE.cap_b, []))
    full.net.step()
    full.check()
    assert full.length() == BASE.cap_b and full.T['c2'].exits == [115.]
    assert full.wait == 0 and full.starts == 1 and full.lost_b == 0

    cases = dict(sparse=replace(BASE, arrival=100, spread=0, jitter_ab=0,
                                jitter_bc=0, warm=0, end=1000),
                 noreserve=replace(BASE, high=26, warm=0, end=30000),
                 early=replace(BASE, high=1, warm=0, end=30000),
                 tight=replace(BASE, cap_b=2, high=3, warm=0, end=30000),
                 nominal=replace(BASE, warm=0, end=30000))
    rows = []
    for name, c in cases.items():
        m = Relay(c, 3303)
        r = m.run(check=True)
        rows.append(dict(case=name, **{k: r[k] for k in (
            'arrivals', 'deliveries', 'loss_a', 'loss_b', 'peak', 'switches',
            'p_on', 'mean_q', 'mean_w')},
            remaining=m.in_a-m.out-m.lost_a-m.lost_b, balance=m.balance()))
    assert rows[0]['arrivals'] == 9 and rows[0]['deliveries'] == 9
    assert rows[0]['mean_w'] == 0 and rows[0]['peak'] == 0
    assert rows[1]['switches'] == 0 and rows[1]['loss_b'] > 0
    assert rows[2]['switches'] > 0 and rows[3]['loss_b'] > 0
    write('verification.csv', rows)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    verify()
    jobs = [(plan, j) for plan in PLANS for j in range(REPS)]
    with Pool() as pool:
        results = pool.map(run_case, jobs, chunksize=1)
        pilots = pool.map(run_pilot, range(PILOTS))
        warm = pool.map(warm_check, range(REPS))
        routes = pool.map(route_check, range(REPS))
    totals, histograms = [], []
    for row, hist in results:
        totals.append(row)
        if row['plan'] == 'H20':
            histograms.extend(dict(rep=row['rep'], n=n, p=p)
                              for n, p in enumerate(hist))
    write('replications.csv', totals)
    write('distribution.csv', histograms)
    write('pilot.csv', [r for rows in pilots for r in rows])
    write('warmup-check.csv', warm)
    write('route-check.csv', routes)
    print(len(totals), 'completed production runs')
