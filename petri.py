"""General simulator of a stochastic timed Petri net (algorithm 3.3).

A net consists of places with integer markings and transitions joined to them by
input, output, inhibitor and test arcs. Each transition has a delay generator,
a priority r and a firing probability b. Transitions are multichannel: every
entry adds one planned exit time to the list E_T of the transition.
"""
from math import inf
from random import Random


class Place:
    def __init__(self, name, mark=0):
        self.name = name
        self.mark = mark

    def __repr__(self):
        return f'{self.name}={self.mark}'


class Transition:
    """Event of the model.

    inputs, outputs -- lists of (place, multiplicity) for ordinary arcs;
    inhibitors -- the transition is enabled only while M(P) < k;
    tests -- the transition needs M(P) >= k but does not remove the tokens;
    delay -- callable returning the delay d >= 0 at the moment of entry;
    priority -- larger value fires first; probability -- weight b in a conflict;
    on_enter, on_exit -- optional callbacks used only for statistics.
    """

    def __init__(self, name, inputs=(), outputs=(), inhibitors=(), tests=(),
                 delay=lambda: 0., priority=0, probability=1.,
                 on_enter=None, on_exit=None):
        self.name = name
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.inhibitors = list(inhibitors)
        self.tests = list(tests)
        self.delay = delay
        self.priority = priority
        self.probability = probability
        self.on_enter = on_enter
        self.on_exit = on_exit
        self.exits = []          # E_T: planned exit times of busy channels

    def enabled(self):
        """Firing condition X(M, T)."""
        return (all(p.mark >= k for p, k in self.inputs)
                and all(p.mark >= k for p, k in self.tests)
                and all(p.mark < k for p, k in self.inhibitors))

    def next_exit(self):
        return min(self.exits, default=inf)

    def __repr__(self):
        return self.name


class PetriNet:
    def __init__(self, places, transitions, conflict_rng=None):
        self.places = {p.name: p for p in places}
        self.transitions = list(transitions)
        self.rng = conflict_rng or Random(0)
        self.t = 0.
        self.trace = None        # callable(kind, transition, net) or None

    def __getitem__(self, name):
        return self.places[name].mark

    def next_event(self):
        """Nearest exit of tokens from any transition."""
        return min(tr.next_exit() for tr in self.transitions)

    # -- exit of tokens: t_{T,r} == t
    def release(self, tr):
        tr.exits.remove(self.t)
        for place, k in tr.outputs:
            place.mark += k
        if tr.on_exit:
            tr.on_exit()
        if self.trace:
            self.trace('exit', tr, self)

    def outputs_phase(self):
        """First loop of algorithm 3.3: all exits planned for the current time.

        Exits only add tokens, so the order of their execution does not change
        the resulting marking.
        """
        for tr in self.transitions:
            while self.t in tr.exits:
                self.release(tr)

    # -- entry of tokens: X(M, T) == 1
    def choose(self):
        """Enabled transition with the highest priority; a conflict between
        transitions of equal priority is resolved randomly with weights b."""
        best, candidates = None, []
        for tr in self.transitions:
            if tr.enabled():
                if best is None or tr.priority > best:
                    best, candidates = tr.priority, [tr]
                elif tr.priority == best:
                    candidates.append(tr)
        if len(candidates) <= 1:
            return candidates[0] if candidates else None
        point = self.rng.random() * sum(tr.probability for tr in candidates)
        for tr in candidates:
            point -= tr.probability
            if point < 0:
                return tr
        return candidates[-1]

    def fire(self, tr):
        for place, k in tr.inputs:
            place.mark -= k
        if tr.on_enter:
            tr.on_enter()
        tr.exits.append(self.t + tr.delay())
        if self.trace:
            self.trace('enter', tr, self)
        # A zero-delay transition passes its tokens at once, so that the next
        # choice already sees its outputs (immediate transition).
        if self.t in tr.exits:
            self.release(tr)

    def inputs_phase(self):
        """Second loop of algorithm 3.3: while some transition is enabled."""
        while (tr := self.choose()) is not None:
            self.fire(tr)

    def step(self):
        self.outputs_phase()
        self.inputs_phase()
