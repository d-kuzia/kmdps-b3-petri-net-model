"""Stochastic Petri net of the B3 packet relay A-B-C; all times in ms.

The net of table 2.2 is described by data (places, arcs, delays, priorities)
and simulated by the general algorithm in petri.py.
"""
from collections import deque
from dataclasses import dataclass, fields
from math import inf
from random import Random
import argparse
import json

from petri import Place, Transition, PetriNet


@dataclass(frozen=True)
class Settings:
    high: int = 20
    band: int = 0
    cap_a: int = 20
    cap_b: int = 25
    arrival: float = 10
    spread: float = 5
    jitter_ab: float = 5
    jitter_bc: float = 3
    fast: float = 15
    warm: float = 20000
    end: float = 220000


class Relay:
    def __init__(self, cfg, seed):
        self.c = cfg
        if not (0 <= cfg.warm < cfg.end and 0 <= cfg.band < cfg.high
                and cfg.cap_a > 0 and cfg.cap_b > 0 and cfg.fast > 0
                and cfg.arrival > cfg.spread >= 0
                and 0 <= cfg.jitter_ab < 20 and 0 <= cfg.jitter_bc < 25):
            raise ValueError("Invalid parameters")
        self.arr, self.route, self.ab_rng, self.bc_rng = [
            Random(10 * seed + j) for j in range(4)]
        self.measured = False
        self.build()
        self.reset_statistics()

    # Net of table 2.2
    def build(self):
        c = self.c
        P = {name: Place(name, mark) for name, mark in [
            ('G', 1), ('X', 0), ('A', 0), ('F_A', c.cap_a), ('L_A', 0),
            ('E1', 1), ('E2', 1), ('U1', 0), ('U2', 0), ('Z1', 0), ('Z2', 0),
            ('B1', 0), ('B2', 0), ('F_B', c.cap_b), ('L_B', 0),
            ('D1', 1), ('D2', 1), ('V1', 0), ('V2', 0), ('C', 0),
            ('R0', 1), ('R1', 0), ('Q', 0)]}
        self.P = P
        # Place Q holds one token per packet waiting in B, so Q >= H is an
        # informational (test) arc Q -> T_on of multiplicity H, and
        # Q < H - band  <=>  M(F_B) >= cap_b + 1 - H + band is an informational
        # arc F_B -> T_off.  Inhibitor arcs are replaced by priorities.
        k_off = c.cap_b + 1 - c.high + c.band
        self.stamps = [deque(), deque()]     # arrival times of tokens in B_i
        self.pending = [0., 0.]              # arrival time of the packet in V_i

        def bc_delay(i):
            def delay():
                if P['R1'].mark:
                    return c.fast
                if i == 0:
                    j = c.jitter_bc
                    return self.bc_rng.uniform(25 - j, 25 + j)
                return 25.
            return delay

        j = c.jitter_ab
        ab_delay = [lambda: 20.,
                    lambda: self.ab_rng.uniform(20 - j, 20 + j)]
        T = [
            Transition('g', [(P['G'], 1)], [(P['G'], 1), (P['X'], 1)],
                       delay=lambda: self.arr.uniform(c.arrival - c.spread,
                                                      c.arrival + c.spread),
                       priority=1, on_exit=self.on_arrival),
            Transition('a', [(P['X'], 1), (P['F_A'], 1)], [(P['A'], 1)],
                       priority=3),
            Transition('l_A', [(P['X'], 1)], [(P['L_A'], 1)], priority=2,
                       on_enter=self.on_loss_a),
            Transition('T_on', [(P['R0'], 1)], [(P['R1'], 1)],
                       tests=[(P['Q'], c.high)], priority=13,
                       on_enter=self.on_switch),
            Transition('T_off', [(P['R1'], 1)], [(P['R0'], 1)],
                       tests=[(P['F_B'], k_off)], priority=13),
        ]
        for i in range(2):
            n = str(i + 1)
            E, U, Z, B, D, V = (P[x + n] for x in 'EUZBDV')
            T += [
                Transition('s' + n, [(P['A'], 1), (E, 1)],
                           [(U, 1), (P['F_A'], 1)], priority=4),
                Transition('u' + n, [(U, 1)], [(Z, 1), (E, 1)],
                           delay=ab_delay[i], priority=14),
                Transition('p' + n, [(Z, 1), (D, 1)], [(V, 1)],
                           priority=11 - (i + 1),
                           on_enter=self.direct(i)),
                Transition('b' + n, [(Z, 1), (P['F_B'], 1)], [(B, 1), (P['Q'], 1)],
                           priority=9 - (i + 1),
                           on_enter=self.buffer(i)),
                Transition('l' + n, [(Z, 1)], [(P['L_B'], 1)],
                           priority=7 - (i + 1), on_enter=self.on_loss_b),
                Transition('v' + n, [(B, 1), (D, 1), (P['Q'], 1)], [(V, 1), (P['F_B'], 1)],
                           priority=13 - (i + 1), on_enter=self.dequeue(i)),
                Transition('c' + n, [(V, 1)], [(P['C'], 1), (D, 1)],
                           delay=bc_delay(i), priority=14,
                           on_enter=self.start_bc(i), on_exit=self.on_delivery),
            ]
        self.T = {tr.name: tr for tr in T}
        self.net = PetriNet(P.values(), T, conflict_rng=self.route)

    # Statistics collected by callbacks of the transitions
    def reset_statistics(self):
        cfg = self.c
        self.in_a = self.lost_a = self.lost_b = self.out = 0
        self.hist = [0.] * (cfg.cap_b + 1)
        self.active = self.wait = self.area_a = 0.
        self.starts = self.fast_starts = self.ons = self.max_q = 0
        self.count = dict(arrivals=0, loss_a=0, offered_b=0, loss_b=0,
                          deliveries=0)
        self.area_lane = [0., 0.]
        self.waits = None                    # optional list of (route, wait)

    def on_arrival(self):
        self.in_a += 1
        self.count['arrivals'] += self.measured

    def on_loss_a(self):
        self.lost_a += 1
        self.count['loss_a'] += self.measured

    def on_loss_b(self):
        self.lost_b += 1
        self.count['offered_b'] += self.measured
        self.count['loss_b'] += self.measured

    def on_switch(self):
        self.ons += self.measured

    def on_delivery(self):
        self.out += 1
        self.count['deliveries'] += self.measured

    def direct(self, i):
        def hook():
            self.count['offered_b'] += self.measured
            self.pending[i] = self.net.t
        return hook

    def buffer(self, i):
        def hook():
            self.count['offered_b'] += self.measured
            self.stamps[i].append(self.net.t)
        return hook

    def dequeue(self, i):
        def hook():
            self.pending[i] = self.stamps[i].popleft()
        return hook

    def start_bc(self, i):
        def hook():
            if self.measured:
                w = self.net.t - self.pending[i]
                self.wait += w
                self.starts += 1
                self.fast_starts += self.P['R1'].mark
                if self.waits is not None:
                    self.waits.append((i, w, self.pending[i]))
        return hook

    def length(self):
        return self.P['B1'].mark + self.P['B2'].mark

    def busy(self):
        return sum(len(self.T[n].exits) for n in ('u1', 'u2', 'c1', 'c2'))

    def balance(self):
        return (self.in_a - self.lost_a - self.lost_b - self.out
                - self.P['A'].mark - self.length() - self.busy())

    def observe(self, stop):
        t = self.net.t
        dt = max(0., stop - max(self.c.warm, t))
        q = self.length()
        self.hist[q] += dt
        self.area_a += dt * self.P['A'].mark
        self.active += dt * self.P['R1'].mark
        self.area_lane[0] += dt * self.P['B1'].mark
        self.area_lane[1] += dt * self.P['B2'].mark
        if dt:
            self.max_q = max(self.max_q, q)

    def check(self):
        P = self.P
        assert self.balance() == 0
        assert P['A'].mark + P['F_A'].mark == self.c.cap_a
        assert self.length() + P['F_B'].mark == self.c.cap_b
        assert P['Q'].mark == self.length()
        assert P['R0'].mark + P['R1'].mark == 1
        assert all(len(self.stamps[i]) == P['B' + str(i + 1)].mark
                   for i in range(2))
        # no enabled transition is left after the entry phase
        assert all(not tr.enabled() for tr in self.net.transitions)

    def run(self, check=False, record=None):
        net = self.net
        if net.t == 0 and not any(tr.exits for tr in net.transitions):
            net.inputs_phase()                # initial entry: g starts
        while net.t < self.c.end:
            stop = min(net.next_event(), self.c.end)
            self.observe(stop)
            net.t = stop
            if stop == self.c.end:
                break
            self.measured = stop >= self.c.warm
            net.step()
            if check:
                self.check()
            if record is not None:
                record(self)
        span = self.c.end - self.c.warm
        assert self.balance() == 0 and abs(sum(self.hist) - span) < 1e-5
        return dict(**self.count, p_on=self.active/span,
                    mean_q=sum(self.area_lane)/span, q1=self.area_lane[0]/span,
                    q2=self.area_lane[1]/span, peak=self.max_q,
                    mean_w=self.wait/max(1, self.starts), starts=self.starts,
                    fast_share=self.fast_starts/max(1, self.starts),
                    switches_s=self.ons*1000/span, switches=self.ons,
                    mean_a=self.area_a/span, p_full=self.hist[-1]/span,
                    rate=self.count['deliveries']*1000/span,
                    hist=[v/span for v in self.hist])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for parameter in fields(Settings):
        option = '--' + parameter.name.replace('_', '-')
        p.add_argument(option, type=parameter.type, default=parameter.default)
    p.add_argument('--seed', type=int, default=3303)
    args = vars(p.parse_args())
    seed = args.pop('seed')
    print(json.dumps(Relay(Settings(**args), seed).run(check=True), indent=2))
