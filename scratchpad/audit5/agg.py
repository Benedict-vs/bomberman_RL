import csv, glob, os, sys, numpy as np
def load(f):
    with open(f) as fh: return list(csv.DictReader(fh))
def agg(pat, agent="benedict_task4"):
    out=[]
    for f in sorted(glob.glob(pat)):
        rs=load(f)
        me=[r for r in rs if r["agent"]==agent]
        opp=[r for r in rs if r["agent"]!=agent]
        g=lambda rows,k: np.array([float(r[k]) for r in rows])
        out.append((os.path.basename(f).replace("__task4_rb_val550731.csv",""), len(me),
            g(me,"score").mean(), g(me,"won").mean(), g(me,"survived").mean(),
            g(me,"suicides").mean(), g(me,"killed_by_opponent").mean(),
            g(me,"crates").mean(), g(me,"coins").mean(), g(me,"kills").mean(),
            g(me,"bombs").mean(), g(me,"steps").mean(),
            g(opp,"survived").mean(), g(opp,"score").mean()))
    return out
if __name__=="__main__":
    hdr=["run","n","score","won","surv","suic","kby","crates","coins","kills","bombs","steps","oppsurv","oppscore"]
    print("  ".join(f"{h:>10}" for h in hdr))
    for row in agg(sys.argv[1]):
        print("  ".join([f"{row[0]:>44}"]+[f"{v:10.3f}" if isinstance(v,float) else f"{v:10d}" for v in row[1:]]))
