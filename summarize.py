"""Print the key estimates and their replicate-based confidence intervals."""
from pathlib import Path
import csv
from statistics import mean, stdev
from math import sqrt

data=Path(__file__).parent/'data'/'replications.csv'
with data.open(encoding='utf-8') as f:
    rows=list(csv.DictReader(f))

def values(plan,column):
    return [float(row[column]) for row in rows if row['plan']==plan]

def estimate(label,observations):
    # 0.975 quantile of Student's t with 24 degrees of freedom
    assert len(observations)==25
    half=2.063898562*stdev(observations)/sqrt(len(observations))
    print(f'{label}: {mean(observations):.6f} +/- {half:.6f}')

for field in ('p_on','mean_q','mean_w','rate','switches_s','fast_share'):
    estimate(field,values('H20',field))
estimate('paired saving (ms)',[a-b for a,b in zip(
    values('confirm20','mean_w'),values('confirm15','mean_w'))])

