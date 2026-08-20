"""Audit 12 / A11: training-time exposure to the post-economy endgame (>= step 200).
E42 dismisses 'less experience' with an aggregate (8-10 % fewer steps); the damage is
confined to steps >= 200, so measure exposure there."""
import csv, numpy as np
def ex(p):
    R=[r for r in csv.DictReader(open(p)) if int(r["episode"])<=20000]
    st=np.array([float(r["steps"]) for r in R])
    return (st>=200).mean(), st[st>=200].sum(), st.sum(), (st>=300).mean()
c=np.array([ex(f"results/train/task4_tournament/benedict_task3__q_e37_PLB2_s{s}.csv") for s in range(100,108)])
m=np.array([ex(f"results/train/task4_tournament/benedict_task4__q_task4_s{s}.csv") for s in range(200,208)])
print(f"{'quantity':<34}{'ctl':>12}{'mix':>12}{'ratio':>9}")
for i,lab in ((0,'P(reaches step 200)'),(3,'P(reaches step 300)'),(2,'total steps'),(1,'steps in episodes >=200')):
    print(f"{lab:<34}{c[:,i].mean():12.3f}{m[:,i].mean():12.3f}{m[:,i].mean()/c[:,i].mean():9.2f}")
