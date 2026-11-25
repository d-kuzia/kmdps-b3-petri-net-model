"""Rebuild report illustrations from experiment.py outputs (matplotlib, pandas)."""
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle
from scipy.interpolate import CubicSpline
from model import Settings, Relay

ROOT=Path(__file__).parent
OUT=ROOT/'media'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
blue='#15476a'
red='#aa4628'
def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.1)
    fig.savefig(OUT/(name+'.png'),dpi=170,bbox_inches='tight',pad_inches=.1)
    plt.close(fig)
def wire(ax, x,y,X,Y):
    ax.add_patch(FancyArrowPatch((x,y),(X,Y),arrowstyle='-|>',mutation_scale=12,
                 color='#314653',linewidth=1.3))
def node(ax,x,y,s):
    ax.text(x,y,s,ha='center',va='center',size=11,bbox=dict(
            boxstyle='round,pad=.38',fc='#eef4f6',ec=blue))
def board(w=10,h=3.6):
    f,a=plt.subplots(figsize=(w,h))
    a.set_xlim(-.7,9.7);a.set_ylim(-1.8,1.8);a.axis('off')
    return f,a
f,a=board()
for x,y,s in [(0,0,'Потік'),(2,0,'A\n20 місць'),(4,1,'AB1'),(4,-1,'AB2'),
              (6,1,'Черга B₁'),(6,-1,'Черга B₂'),(8,1,'BC1'),(8,-1,'BC2'),(9.3,0,'C')]:
    node(a,x,y,s)
wire(a,.48,0,1.45,0)
for y in (1,-1):
    wire(a,2.65,0,3.55,y);wire(a,4.45,y,5.35,y);wire(a,6.65,y,7.55,y)
    wire(a,8.42,y,9.1,0)
a.text(6,0,'Разом 25 місць',ha='center',size=9)
save(f,'system')
# Petri net drawings in the notation of table 3.2 and figure 3.2 of the guide:
# a circle is a place (the number inside is the marking, the name is above),
# a rounded bar is a transition with the event name above and, below it, the
# delay d and the pair "r, b" (numbers as in the program, with a decimal
# point); a dashed arrow is an informational arc, which needs k tokens in its
# place but does not remove them.
PR, TW, TH = .33, .11, .37
FN, FP, FK, FT = 14, 11.5, 12, 12   # font sizes: place names, parameters, multiplicities, events


class Sketch:
    def __init__(self, ax):
        self.ax, self.nodes = ax, {}
        ax.set_aspect('equal')
        ax.axis('off')

    def place(self, key, x, y, label, mark=0, at=(0, .52)):
        self.ax.add_patch(Circle((x, y), PR, fc='white', ec='black', lw=1.0))
        if label:
            self.ax.text(x+at[0], y+at[1], label, ha='center', va='center', fontsize=FN)
        if mark:
            self.ax.text(x, y, str(mark), ha='center', va='center', fontsize=FP)
        self.nodes[key] = ('p', x, y)

    def trans(self, key, x, y, label, d, r, b='1.0', at=(0, 0), par=(0, 0)):
        self.ax.add_patch(FancyBboxPatch((x-TW, y-TH), 2*TW, 2*TH, fc='white',
                                         ec='black', lw=1.0,
                                         boxstyle='round,pad=0,rounding_size=.07'))
        self.ax.text(x+at[0], y+TH+.08+at[1], label, ha='center', va='bottom',
                     fontsize=FT, linespacing=1.05)
        self.ax.text(x+par[0], y-TH-.06+par[1], f'{d}\n{r}, {b}', ha='center',
                     va='top', fontsize=FP, linespacing=1.1)
        self.nodes[key] = ('t', x, y)

    def inside(self, key, px, py):
        kind, x, y = self.nodes[key]
        if kind == 'p':
            return (px-x)**2 + (py-y)**2 <= (PR+.02)**2
        return abs(px-x) <= TW+.02 and abs(py-y) <= TH+.02

    def arc(self, a, b, bend=0., kind='arc', k='', kpos=.5, kside=1, kdist=.24, via=()):
        _, x1, y1 = self.nodes[a]
        _, x2, y2 = self.nodes[b]
        if via:
            pts = np.array([(x1, y1), *via, (x2, y2)], float)
            u = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
            ts = np.linspace(0, u[-1], 1200)
            xs, ys = CubicSpline(u, pts[:, 0])(ts), CubicSpline(u, pts[:, 1])(ts)
        else:
            mx, my = (x1+x2)/2, (y1+y2)/2
            cx, cy = mx - bend*(y2-y1), my + bend*(x2-x1)
            ts = np.linspace(0, 1, 600)
            xs = (1-ts)**2*x1 + 2*(1-ts)*ts*cx + ts**2*x2
            ys = (1-ts)**2*y1 + 2*(1-ts)*ts*cy + ts**2*y2
        keep = [i for i in range(len(xs))
                if not self.inside(a, xs[i], ys[i]) and not self.inside(b, xs[i], ys[i])]
        i0, i1 = keep[0], keep[-1]
        step = np.hypot(np.diff(xs), np.diff(ys)).mean()
        back = max(3, int(.1/step))
        if kind == 'inhibitor':
            ex, ey = xs[i1], ys[i1]
            dx, dy = xs[i1]-xs[i1-back], ys[i1]-ys[i1-back]
            n = np.hypot(dx, dy)
            r = .075
            ccx, ccy = ex - dx/n*r, ey - dy/n*r
            j = max(i for i in range(i0, i1) if (xs[i]-ccx)**2+(ys[i]-ccy)**2 >= r*r)
            self.ax.plot(xs[i0:j+1], ys[i0:j+1], color='black', lw=.95)
            self.ax.add_patch(Circle((ccx, ccy), r, fc='white', ec='black', lw=.95))
        else:
            self.ax.plot(xs[i0:i1-back//2], ys[i0:i1-back//2], color='black', lw=.95,
                         ls=(0, (4, 3)) if kind == 'info' else '-')
            self.ax.add_patch(FancyArrowPatch((xs[i1-back], ys[i1-back]), (xs[i1], ys[i1]),
                              arrowstyle='-|>', mutation_scale=11, color='black',
                              lw=.95, shrinkA=0, shrinkB=0))
        if k:
            # multiplicity of the arc written beside it
            j = int(i0 + (i1-i0)*kpos)
            dx, dy = xs[j+3]-xs[j-3], ys[j+3]-ys[j-3]
            n = np.hypot(dx, dy)
            nx, ny = -dy/n, dx/n
            self.ax.text(xs[j]+kside*kdist*nx, ys[j]+kside*kdist*ny, k, ha='center',
                         va='center', fontsize=FK)


def ev(word, m, sub=''):
    """Event name in words with the short name of the transition in brackets."""
    return f'{word} ({nm(m, sub)})'


def nm(m, sub=''):
    """Upright name with an optional subscript, as on figure 3.2 of the guide."""
    m = m if m.startswith('\\') else rf'\mathrm{{{m}}}'
    if not sub:
        return f'${m}$'
    sub = 'i' if sub == 'i' else rf'\mathrm{{{sub}}}'
    return f'${m}_{{{sub}}}$'


# The Petri net schemes use the serif face of the report text (Times New Roman
# metrics), as figure 3.2 of the guide does; the charts keep the sans face.
SANS = {k: plt.rcParams[k] for k in ('font.family', 'mathtext.fontset', 'mathtext.rm',
                                     'mathtext.it', 'mathtext.bf', 'mathtext.fallback')}
plt.rcParams.update({'font.family': 'Liberation Serif', 'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Liberation Serif', 'mathtext.it': 'Liberation Serif:italic',
                     'mathtext.bf': 'Liberation Serif:bold', 'mathtext.fallback': 'stix'})

# Fig. 2.1: generator, buffer A and line AB_i
f, a = plt.subplots(figsize=(10.4, 3.25))
s = Sketch(a)
s.place('G', -1.8, 0, nm('G'), 1)
s.trans('g', 0, 0, ev('Надходження\n', 'g'), 'unif(10, 5)', '1')
s.place('X', 1.8, 0, nm('X'), at=(-.42, .4))
s.trans('a', 3.6, 0, ev('Прийом\nв A', 'a'), '0', '3', at=(-.42, 0))
s.place('A', 5.4, 0, nm('A'), at=(-.4, .42))
s.trans('s', 7.2, 0, ev('Старт\nAB$_i$', 's', 'i'), '0', '4', '0.5', at=(-1.4, 0))
s.place('U', 9, 0, nm('U', 'i'), at=(.42, .42))
s.trans('u', 10.8, 0, ev('Передача\nAB$_i$', 'u', 'i'), 'τ(ABᵢ)', '14', at=(.9, 0))
s.place('Z', 13.0, 0, nm('Z', 'i'))
s.trans('lA', 1.8, 2.3, ev('Відмова\nв A', r'\ell', 'A'), '0', '2', par=(-.82, .2))
s.place('LA', 0, 2.3, nm('L', 'A'))
s.place('FA', 5.4, 2.3, nm('F', 'A'), 20)
s.place('E', 9, 2.3, nm('E', 'i'), 1)
for x, y in [('X', 'a'), ('a', 'A'), ('A', 's'), ('s', 'U'), ('U', 'u'),
             ('u', 'Z'), ('X', 'lA'), ('lA', 'LA'), ('FA', 'a'), ('s', 'FA'),
             ('E', 's'), ('u', 'E'), ('g', 'X')]:
    s.arc(x, y)
s.arc('G', 'g', bend=.2)
s.arc('g', 'G', bend=.2)
a.set(xlim=(-2.4, 13.6), ylim=(-1.3, 3.55))
save(f, 'petri_a')

# Fig. 2.2: buffer B, line BC_i (top) and the reserve controller (bottom)
f, (a, b) = plt.subplots(2, 1, figsize=(9.2, 9.0), gridspec_kw={'height_ratios': [7.0, 5.6], 'hspace': .08})
s = Sketch(a)
s.place('Z', 0, 0, nm('Z', 'i'), at=(-.45, .42))
s.trans('b', 1.8, 0, ev('У чергу\nB$_i$', 'b', 'i'), '0', '9−i', par=(-.95, 0), at=(-.18, 0))
s.place('B', 3.6, 0, nm('B', 'i'))
s.trans('v', 5.4, 0, ev('З черги\nна BC$_i$', 'v', 'i'), '0', '13−i', par=(1.3, -.75), at=(.2, 0))
s.place('V', 7.2, 0, nm('V', 'i'), at=(-.4, .4))
s.trans('c', 9, 0, ev('Передача\nBC$_i$', 'c', 'i'), 'τ(BCᵢ)', '14', par=(.68, 0))
s.place('C', 10.8, 0, nm('C'))
s.trans('p', 7.2, 3.0, ev('Напряму\nна BC$_i$', 'p', 'i'), '0', '11−i', par=(1.2, .6))
s.place('D', 9, -2.2, nm('D', 'i'), 1, at=(.55, -.1))
s.trans('l', 0, -2.2, ev('Відмова\nв B', r'\ell', 'i'), '0', '7−i', at=(-.75, 0), par=(.85, .2))
s.place('LB', -1.8, -2.2, nm('L', 'B'))
s.place('FB', 3.6, -1.5, nm('F', 'B'), 25, at=(.55, -.25))
s.place('Q', 3.6, -3.1, nm('Q'), at=(.5, -.2))
for x, y in [('Z', 'b'), ('b', 'B'), ('B', 'v'), ('v', 'V'), ('V', 'c'),
             ('c', 'C'), ('D', 'p'), ('p', 'V'), ('D', 'v'), ('c', 'D'),
             ('Z', 'l'), ('l', 'LB'), ('FB', 'b'), ('v', 'FB')]:
    s.arc(x, y)
s.arc('Z', 'p', via=[(.9, 1.75), (3.6, 2.95)])
s.arc('b', 'Q', via=[(1.9, -2.2)])
s.arc('Q', 'v', via=[(5.3, -2.2)])
a.set(xlim=(-2.3, 11.3), ylim=(-3.6, 4.15))
s = Sketch(b)
s.place('R0', 0, 0, nm('R', '0'), 1, at=(-.68, 0))
s.place('R1', 3.2, 0, nm('R', '1'), at=(.68, 0))
s.trans('on', 1.6, 1.7, ev('Увімкнення\nрезерву', 'T', 'on'), '0', '13', par=(.95, .75))
s.trans('off', 1.6, -1.7, ev('Вимкнення\nрезерву', 'T', 'off'), '0', '13', at=(0, -1.98), par=(.95, .3))
s.place('Q', -1.6, 1.7, nm('Q'))
s.place('FB', -1.6, -1.7, nm('F', 'B'), 25)
s.arc('R0', 'on'); s.arc('on', 'R1')
s.arc('R1', 'off'); s.arc('off', 'R0')
s.arc('Q', 'on', kind='info', k='H', kpos=.4, kdist=.3)
s.arc('FB', 'off', kind='info', k='26−H+δ', kpos=.4, kside=-1, kdist=.3)
b.set(xlim=(1.0-6.8, 1.0+6.8), ylim=(-3.4, 2.75))
save(f, 'petri_b')
plt.rcParams.update(SANS)
# UML class diagram: name / attributes / operations compartments, composition
# (filled diamond at the whole), association and a dashed dependency.
f,a=plt.subplots(figsize=(10.4,5.6))
a.set(xlim=(-.1,11.3),ylim=(-1.05,4.75))
a.axis('off')
ROW, HEAD, PAD = .30, .42, .14
def class_box(x,y,w,name,fields,methods,stereo=''):
    hf=PAD+ROW*len(fields)+PAD/2
    hm=PAD+ROW*len(methods)+PAD/2 if methods else .22
    head=HEAD+(.24 if stereo else 0)
    h=head+hf+hm
    a.add_patch(Rectangle((x,y-h),w,h,facecolor='white',edgecolor=blue,lw=1.4))
    if stereo:
        a.text(x+w/2,y-.18,stereo,ha='center',va='center',size=8.5,color=blue)
    a.text(x+w/2,y-head+.2,name,ha='center',va='center',weight='bold',color=blue,size=10)
    a.plot([x,x+w],[y-head]*2,color=blue,lw=.9)
    for j,t in enumerate(fields):
        a.text(x+.09,y-head-PAD-ROW*(j+.5),t,va='center',size=8.8)
    sep=y-head-hf
    a.plot([x,x+w],[sep]*2,color=blue,lw=.9)
    for j,t in enumerate(methods):
        a.text(x+.09,sep-PAD-ROW*(j+.5),t,va='center',size=8.8)
    return x,y-h,w,h
def diamond(x,y,d):
    a.add_patch(plt.Polygon([(x,y),(x+d*.16,y+.09),(x+d*.32,y),(x+d*.16,y-.09)],
                            closed=True,fc=blue,ec=blue,zorder=3))
def compose(xw,xp,y,mult):
    # whole at xw, part at xp (horizontal line)
    d=-1 if xw>xp else 1
    diamond(xw,y,d)
    a.plot([xw+d*.32,xp],[y,y],color=blue,lw=1.1)
    a.text(xp-d*.08,y+.12,mult,ha='left' if d<0 else 'right',size=8.5)
class_box(.0,4.6,2.55,'Place',['+name: str','+mark: int'],['+__repr__(): str'])
class_box(.0,2.3,2.55,'Transition',['+inputs, outputs: list','+inhibitors, tests: list','+delay: callable','+priority: int','+probability: float','+exits: list[float]'],
          ['+enabled(): bool','+next_exit(): float'])
class_box(3.35,4.6,2.6,'PetriNet',['+places: dict[str, Place]','+transitions: list','+t: float','+rng: Random'],
          ['+next_event(): float','+outputs_phase()','+inputs_phase()','+choose(): Transition','+fire(tr), release(tr)','+step()'])
class_box(6.6,4.6,2.4,'Relay',['+c: Settings','+P, T: dict','+net: PetriNet','+stamps: list[deque]'],
          ['+build()','+run(check, record): dict','+observe(stop)','+check(), balance()'])
class_box(9.4,4.6,1.85,'Settings',['+high, band: int','+cap_a, cap_b: int','+arrival, spread','+jitter_ab, jitter_bc','+fast: float','+warm, end: float'],[],stereo='«dataclass»')
class_box(9.4,1.6,1.85,'experiment',[],['+run_case(job)','+run_pilot(j)','+verify()'],stereo='«module»')
compose(3.35,2.55,4.0,'1..*')     # PetriNet <>-- Place
compose(3.35,2.55,1.9,'1..*')     # PetriNet <>-- Transition
compose(6.6,5.95,4.0,'1')         # Relay <>-- PetriNet
# Transition -- Place: arcs (place, multiplicity)
a.plot([1.27,1.27],[2.3,2.86],color=blue,lw=1.1)
a.text(1.38,2.58,'дуги (P, k)',va='center',size=8.5)
a.text(1.36,2.70,'*',size=8.5); a.text(1.36,2.33,'*',size=8.5)
# Relay --> Settings (navigable association)
a.add_patch(FancyArrowPatch((9.0,4.0),(9.4,4.0),arrowstyle='->',mutation_scale=11,color=blue,lw=1.1))
# experiment ..> Relay: dependency «create»
a.add_patch(FancyArrowPatch((9.4,1.0),(7.72,1.45),arrowstyle='->',mutation_scale=11,color=blue,lw=1.1,linestyle='--',
                            connectionstyle='angle,angleA=180,angleB=90'))
a.text(8.6,.82,'«create»',ha='center',size=8.5,color=blue)
save(f,'classes')
x=pd.read_csv(ROOT/'data/replications.csv')
p=pd.read_csv(ROOT/'data/pilot.csv')
h=pd.read_csv(ROOT/'data/distribution.csv')
import json
res=json.load(open(ROOT/'data/analysis.json',encoding='utf-8'))
f,(a,b)=plt.subplots(1,2,figsize=(9.6,3.4))
for rep,g in p.groupby('rep'):
    a.plot(g.t/1000,g.q,lw=1);b.plot(g.t/1000,100*g.reserve,lw=1)
a.set(xlabel='t, с',ylabel='Середня черга B на [0; t], пак.',ylim=(14,19.6))
b.set(xlabel='t, с',ylabel='Частка часу резерву на [0; t], %',ylim=(8,18))
a.axvline(res['transient']['q']['T']/1000,ls=':',color='k')
b.axvline(res['transient']['reserve']['T']/1000,ls=':',color='k')
for ax in (a,b): ax.grid(alpha=.22);ax.set_xlim(0,220)
f.tight_layout();save(f,'pilot')
f,a=plt.subplots(figsize=(7,3.3))
g=h.groupby('n').p.mean().loc[16:22]*100
a.bar(g.index,g,color=[red if v>=20 else blue for v in g.index])
a.set(xlabel='Черга в B, пакетів',ylabel='Частка часу, %')
for n,v in g.items():
    if v>0:a.text(n,v+.9,f'{v:.2f}'.replace('.',','),ha='center')
a.set_ylim(0,85);f.tight_layout();save(f,'hist')
f,(a,b)=plt.subplots(1,2,figsize=(8.9,3.4))
z=x[x.plan.str.match(r'^H\d+$')].copy();z['H']=z.plan.str[1:].astype(int)
g=z.groupby('H')
a.plot(g.mean_w.mean().index,g.mean_w.mean(),'o',color=blue)
rg=res['regression'];hh=np.linspace(10,26,50)
a.plot(hh,rg['b0']+rg['b1']*hh,'--',color='#666',lw=1)
ratio=100*g.loss_b.sum()/g.offered_b.sum()
b.plot(ratio.index,ratio,'o-',color=red)
for ax,ylab in [(a,'Очікування в B, мс'),(b,'Частка втрат у B, %')]:
    ax.set_xticks(ratio.index,ratio.index,rotation=55)
    ax.set(xlabel='Поріг H, пакетів',ylabel=ylab);ax.grid(alpha=.22)
f.tight_layout();save(f,'threshold')
f,(a,b)=plt.subplots(2,1,figsize=(7.6,3.7),sharex=True,
                      gridspec_kw={'height_ratios':[2,1]})
history=[]
def record(m):
    if m.net.t>=4200: history.append((m.net.t,m.length(),m.P['R1'].mark))
Relay(replace(Settings(),warm=0,end=4500),3303).run(record=record)
t,q,r=zip(*history)
a.step(t,q,where='post',color=blue);b.step(t,r,where='post',color=red)
a.axhline(20,ls=':',color='k');a.set_ylabel('Черга, пакетів')
a.set_yticks(range(min(q),max(q)+1))
b.set_yticks([0,1],['0 — вимкнено','1 — увімкнено'])
b.set(yticks=[0,1],ylabel='Режим',xlabel='Час моделі, мс')
f.tight_layout();save(f,'trace')

grid = pd.read_csv(ROOT/'data/wait-cdf.csv')
f,a=plt.subplots(figsize=(7.3,3.5))
for key,label,color in [('all','Обидва маршрути',blue),('BC1','BC1',red),('BC2','BC2','#58815b')]:
    g=grid[grid.route==key].groupby('ms')[['n','passed']].sum()
    a.plot(g.index,100*g.passed/g.n,label=label,color=color)
a.set(xlabel='Час очікування у B, мс',ylabel='Частка пакетів, %',ylim=(0,103))
a.grid(alpha=.22);a.legend(frameon=False);f.tight_layout();save(f,'waitcdf')

ch=res['chi2'];u=ch['tests']['uniform'];nm=ch['tests']['normal']
e=np.array(u['edges']);obs=np.array(u['observed']);mid=(e[1:]+e[:-1])/2;wd=e[1]-e[0]
f,a=plt.subplots(figsize=(7.3,3.3))
a.bar(mid,obs,width=wd*.92,color=blue,alpha=.75,label='Спостережені частоти')
a.plot(mid,u['expected'],'o-',color=red,label='Рівномірний закон')
a.plot(mid,nm['expected'],'s--',color='#58815b',ms=4,label='Нормальний закон')
a.set(xlabel='Час очікування у B, мс',ylabel='Кількість пакетів')
a.grid(alpha=.22);a.legend(frameon=False,fontsize=9);f.tight_layout();save(f,'chi2')
