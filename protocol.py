"""Print fragments of the event protocol: every exit and entry of tokens with
the marking of buffer B, the controller and the planned exits of BC lines."""
from dataclasses import replace
from pathlib import Path
from model import Relay, Settings
from experiment import state, BASE

OUT = Path(__file__).parent/'data'


def show(m, kind, tr, net):
    P = m.P
    c1 = ','.join(f'{v:.3f}' for v in m.T['c1'].exits) or '-'
    c2 = ','.join(f'{v:.3f}' for v in m.T['c2'].exits) or '-'
    word = 'вихід' if kind == 'exit' else 'вхід '
    return (f'{net.t:8.3f} {word} {tr.name:5s} A={P["A"].mark:<2d} B1={P["B1"].mark:<2d} '
            f'B2={P["B2"].mark:<2d} FB={P["F_B"].mark:<2d} R1={P["R1"].mark} '
            f'c1={c1:<8s} c2={c2}')


def protocol(m, start, stop):
    lines = []
    m.net.trace = lambda kind, tr, net: (
        lines.append(show(m, kind, tr, net)) if start <= net.t <= stop else None)
    return lines


if __name__ == '__main__':
    head = f'{"t, мс":>8s} подія перехід, маркірування після події, заплановані виходи c1, c2'
    out = []
    # 1. Nominal trajectory, seed 3303: the first switch of the reserve.
    m = Relay(replace(Settings(), warm=0, end=1300), 3303)
    first = []
    m.net.trace = lambda kind, tr, net: (
        first.append(net.t) if tr.name == 'T_on' and kind == 'enter' else None)
    m.run()
    t_on = first[0]
    m = Relay(replace(Settings(), warm=0, end=t_on + 40), 3303)
    lines = protocol(m, t_on - 2.3, t_on + 17.0)
    m.run(check=True)
    out += [f'Фрагмент 1. Перше ввімкнення резерву, початкове значення 3303', head]
    out += lines
    # 2. Prepared state: c1 and c2 finish together at 100 ms, 10 + 10 tokens
    #    in B, reserve on.
    m = Relay(BASE, 3303)
    state(m, 100., dict(G=0, B1=10, B2=10, F_B=5, D1=0, D2=0, R0=0, R1=1),
          dict(g=[120.], c1=[100.], c2=[100.]), ([90.]*10, [90.]*10))
    lines = protocol(m, 0, 1e9)
    m.net.step()
    out += ['', 'Фрагмент 2. Одночасне завершення c1 і c2 при 20 пакетах у B', head]
    out += lines
    text = '\n'.join(out) + '\n'
    (OUT/'protocol.txt').write_text(text, encoding='utf-8')
    print(text)
