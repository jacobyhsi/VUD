from pathlib import Path
import pandas as pd
import numpy as np
import json
root=Path("/home/js2723/PROJECTS/VUD/results")
def read(folder):
    rows=[]
    for p in folder.glob("results_test_x*.csv"):
        d=pd.read_csv(p)
        cols=[c for c in d if c.startswith("x_") and c!="x_note"]
        kl=d.kl_pyx_pyxz
        sel=d.loc[kl.isin(kl.nsmallest(5))]
        raw=sel.Va.min()
        tu=d["H[p(y|x,D)]"].iloc[0]
        r={c:float(d[c].iloc[0]) for c in cols}
        r.update(TU=tu,Va=min(raw,tu),rawVa=raw,clipped=raw>tu,KL=sel.loc[sel.Va.idxmin(),"kl_pyx_pyxz"],bad=int(d[["Va","H[p(y|x,D)]"]].isna().any(axis=1).sum()))
        r["Ve"]=tu-r["Va"]
        if "n_joint_samples" in d:r["N"]=d.n_joint_samples.iloc[0]
        if "pp_covariance_regularized" in d:r["regularized"]=int(d.pp_covariance_regularized.sum())
        if "p(y|x,D)_mean" in d:r["prediction"]=d["p(y|x,D)_mean"].iloc[0]
        rows.append(r)
    return pd.DataFrame(rows)
for task,vpath,ppath in [
("gaps",root/"toy_regression/gaps/Dream-v0-Base-7B/va_vanilla/seed_0",root/"toy_regression/gaps/Dream-v0-Base-7B/va_parallel_pp/seed_0"),
("linear",root/"toy_regression/linear_regression/Dream-v0-Base-7B/va_vanilla/seed_0",root/"toy_regression/linear_regression/Dream-v0-Base-7B/va_parallel_pp/seed_0"),
("spirals",root/"toy_classification/spirals/Dream-v0-Base-7B/va_vanilla",root/"toy_classification/spirals/Dream-v0-Base-7B/va_parallel_pp")]:
    v,p=read(vpath),read(ppath)
    keys=[c for c in v if c.startswith("x_")]
    m=v.merge(p,on=keys,suffixes=("_v","_p"),validate="one_to_one")
    print("\nTASK",task,"counts",len(v),len(p),"matched",len(m),"range",m[keys].agg(["min","max"]).to_dict())
    print("same ICL",pd.read_csv(vpath/"D_test.csv").equals(pd.read_csv(ppath/"D_test.csv")))
    print("N",p.N.unique() if "N" in p else None)
    for suffix in ["v","p"]:
        print(suffix,{c:round(float(m[c+"_"+suffix].mean()),4) for c in ["TU","Va","Ve","rawVa","clipped","KL","bad"]})
    print("Va corr",m.Va_v.corr(m.Va_p),"PP Va higher fraction",(m.Va_p>m.Va_v).mean())
    if task!="spirals":
        x=keys[0]
        for lo,hi in ([(-9,-5),(-3,1),(3,7)] if task=="gaps" else [(-15,-3),(-3,5),(5,15)]):
            sub=m[m[x].between(lo,hi)]
            print("region",lo,hi,"n",len(sub),{c:round(float(sub[c].mean()),4) for c in ["TU_v","Va_v","Ve_v","TU_p","Va_p","Ve_p"]})
        if task=="linear":
            truth=3-m[x]
            for s in ["v","p"]:
                print("prediction RMSE",s,float(np.sqrt(np.mean((m["prediction_"+s]-truth)**2))))
    print("PP full",p[["TU","Va","Ve"]].mean().to_dict())
