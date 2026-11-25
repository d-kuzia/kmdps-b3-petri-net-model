"""Audit the B queue integral and describe the distribution of packet wait times."""
from math import ceil
from bisect import bisect_right
from multiprocessing import Pool
from model import Relay, Settings
from experiment import write, OUT, REPS

THIN = 500         # every THIN-th start enters the sample for the chi-square test


def audit(j):
    seed = 33003 + 97*j
    m = Relay(Settings(), seed)
    m.waits = []
    summary = m.run()
    c = m.c
    samples = [[w for i, w, _ in m.waits if i == r] for r in (0, 1)]
    waits = [w for _, w, _ in m.waits]
    left = sum(max(0., c.warm - a) for _, _, a in m.waits)
    tail = sum(c.end - max(stamp, c.warm) for lane in m.stamps for stamp in lane)
    area_packets = sum(waits) - left + tail
    area_events = sum(m.area_lane)
    assert abs(area_packets - area_events) < 1e-4
    assert len(waits) == summary['starts']
    rows, grid = [], []
    for route, sample in [('BC1', samples[0]), ('BC2', samples[1]), ('all', waits)]:
        ordered = sorted(sample)
        n = len(ordered)
        rows.append(dict(rep=j+1, seed=seed, route=route, starts=n,
                         mean=sum(ordered)/n,
                         sd=(sum((w - sum(ordered)/n)**2 for w in ordered)/(n-1))**.5,
                         over_250=sum(w > 250 for w in ordered),
                         p50=ordered[ceil(.50*n)-1], p95=ordered[ceil(.95*n)-1],
                         p99=ordered[ceil(.99*n)-1], peak=max(ordered)))
        for limit in range(0, 451, 10):
            grid.append(dict(rep=j+1, route=route, ms=limit, n=n,
                             passed=bisect_right(ordered, limit)))
    check = dict(rep=j+1, seed=seed, events=area_events, packets=area_packets,
                 left=left, right=tail, residual=area_packets-area_events)
    thinned = [dict(rep=j+1, k=k, route=i+1, wait=w)
               for k, (i, w, _) in enumerate(m.waits) if k % THIN == THIN-1]
    return rows, check, grid, thinned


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    with Pool() as pool:
        parts = pool.map(audit, range(REPS))
    write('waiting.csv', [r for p in parts for r in p[0]])
    write('area-check.csv', [p[1] for p in parts])
    write('wait-cdf.csv', [r for p in parts for r in p[2]])
    write('wait-sample.csv', [r for p in parts for r in p[3]])
