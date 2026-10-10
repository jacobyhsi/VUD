from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from csaps import csaps
ROOT=Path("/home/js2723/PROJECTS/VUD")
OUT=ROOT/"results/comparisons/pp_vs_vanilla"
OUT.mkdir(parents=True,exist_ok=True)
def read(folder):
    rows=[]
    for path in sorted(folder.glob("results_test_x*.csv")):
        d=pd.read_csv(path)
        k=d.kl_pyx_pyxz
        s=d.loc[k.isin(k.nsmallest(5))]
        raw=s.Va.min()
        tu=d["H[p(y|x,D)]"].iloc[0]
        r={c:float(d[c].iloc[0]) for c in d if c.startswith("x_") and c!="x_note"}
        r.update(TU=tu,Va=min(raw,tu),rawVa=raw,Ve=tu-min(raw,tu))
        rows.append(r)
    return pd.DataFrame(rows)
def curve(ax,x,y,color,label,style="-"):
    x=np.asarray(x); y=np.asarray(y)
    valid=np.isfinite(y)
    ax.scatter(x[valid],y[valid],s=12,color=color,alpha=.25)
    # Smooth only contiguous finite segments; do not bridge missing values.
    groups=np.split(np.flatnonzero(valid), np.flatnonzero(np.diff(np.flatnonzero(valid))>1)+1)
    first=True
    for ids in groups:
        if len(ids)<2: continue
        grid=np.linspace(x[ids].min(),x[ids].max(),max(20,len(ids)*2))
        ax.plot(grid,csaps(x[ids],y[ids],smooth=.85)(grid),style,color=color,lw=2,label=label if first else None)
        first=False
    if (~valid).any():
        ax.plot(x[~valid],np.full((~valid).sum(),.03),"rx",transform=ax.get_xaxis_transform(),label="Non-finite estimate")
for name,dataset,vs,ps in [
("gaps","toy_regression/gaps","va_vanilla/seed_0","va_parallel_pp/seed_0"),
("linear","toy_regression/linear_regression","va_vanilla/seed_0","va_parallel_pp/seed_0"),
("spirals","toy_classification/spirals","va_vanilla","va_parallel_pp")]:
    base=ROOT/"results"/dataset/"Dream-v0-Base-7B"
    v,p=read(base/vs),read(base/ps)
    keys=[c for c in v if c.startswith("x_")]
    m=v.merge(p,on=keys,suffixes=("_v","_p"),validate="one_to_one").sort_values(keys)
    m.to_csv(OUT/f"{name}_matched.csv",index=False)
    if name=="spirals":
        fixed=keys[0]; x=keys[1]
        # Compare the first sampled column only, with its actual available coverage.
        m=m.loc[m[fixed].eq(m[fixed].min())]
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        for ax,col in zip(axes,["TU","Va","Ve"]):
            curve(ax,m[x],m[col+"_v"],"C0","Vanilla")
            curve(ax,m[x],m[col+"_p"],"C1","Parallel-PP")
            ax.set(title=col,xlabel="Test covariate x2",ylabel="Uncertainty (bits)")
            ax.legend(); ax.grid(alpha=.15)
        fig.suptitle(f"Spirals: matched edge only, x1={m[fixed].iloc[0]:g}; {len(m)} points (not the full surface)")
    else:
        x=keys[0]
        fig,axes=plt.subplots(2,3,figsize=(16,9))
        for ax,d,label in [(axes[0,0],v,"Vanilla"),(axes[0,1],p,"Parallel-PP")]:
            d=d.sort_values(x)
            curve(ax,d[x],d.TU,"C0","Total")
            curve(ax,d[x],d.Va,"C1","Aleatoric (capped)")
            for i,dx in enumerate(pd.read_csv(base/(vs if label=="Vanilla" else ps)/"D_test.csv")[x[2:]]):
                ax.axvline(dx,color="C5",ls="--",alpha=.3,label="ICL data" if i==0 else None)
            ax.set(title=f"{label}: {len(d)}/150 points",xlim=(-15,15))
            if label=="Parallel-PP": ax.axvspan(m[x].max(),15,color="grey",alpha=.12,label="No vanilla comparison yet")
        for ax,col,title in [(axes[0,2],"TU","Matched total uncertainty"),(axes[1,0],"Va","Matched aleatoric (capped)"),(axes[1,1],"Ve","Matched epistemic (TU - capped Va)"),(axes[1,2],"rawVa","Matched aleatoric BEFORE cap")]:
            curve(ax,m[x],m[col+"_v"],"C0","Vanilla")
            curve(ax,m[x],m[col+"_p"],"C1","Parallel-PP")
            ax.set_title(title)
            if name=="linear" and col in ("Va","rawVa"):
                ax.axhline(.5*np.log(2*np.pi*np.e*4),color="black",ls=":",label="True noise entropy")
        for ax in axes.flat:
            ax.set_xlabel("Test covariate x"); ax.set_ylabel("Uncertainty (nats)")
            ax.legend(fontsize=8); ax.grid(alpha=.15)
        # Same scale for the two full decomposition panels.
        lims=[axes[0,i].get_ylim() for i in range(2)]
        for ax in axes[0,:2]: ax.set_ylim(min(z[0] for z in lims),max(z[1] for z in lims))
        fig.suptitle(f"{name.capitalize()}: seed 0, {len(m)} matched points; smooth=0.85; red crosses = non-finite values")
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(OUT/f"{name}_comparison.png",dpi=150)
    plt.close(fig)
    print(name,len(v),len(p),len(m),flush=True)
