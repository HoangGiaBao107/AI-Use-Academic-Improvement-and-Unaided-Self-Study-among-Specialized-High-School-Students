"""Paired respondent bootstrap and the C3 parametric-null check."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from threadpoolctl import threadpool_limits
from .data import load_analysis
from .firth import fit_firth,test_firth
from .storage import save_table,finish_phase,write_json

PHASE='04_uncertainty'
SEEDS={'CAU':2026102614,'C2':20261026,'C3':2026102615,'Components':20261107}
TAGS={'CAU':14,'C2':1,'C3':15,'Components':17}
WORK={}

def initialize(jobs,cau,aor):
    threadpool_limits(1)
    WORK.update(jobs=jobs,cau=cau,aor=aor)

def bootstrap_chunk(reps):
    rows=[];correlations=[]
    for rep in reps:
        weights={};indices={}
        for group in SEEDS:
            rng=np.random.default_rng(np.random.SeedSequence([SEEDS[group],rep,TAGS[group]]))
            ix=rng.integers(323,size=323);indices[group]=ix
            weights[group]=np.bincount(ix,minlength=323).astype(float)
        correlations.append(dict(replicate=rep,rho=float(stats.spearmanr(
            WORK['cau'][indices['Components']],WORK['aor'][indices['Components']]).statistic)))
        for key,job in WORK['jobs'].items():
            group,scenario,outcome=key
            p=job['payload'];row=dict(group=group,scenario=scenario,outcome=outcome,replicate=rep,valid=False)
            try:
                w=weights[group][job['indices']]
                fit=fit_firth(p['X'],job['y'],w,job['fit']['beta'],protected=p['protected'])
                low=float(np.average(expit(job['Xlow']@fit['beta']),weights=w))
                high=float(np.average(expit(job['Xhigh']@fit['beta']),weights=w))
                row.update(valid=True,n_sampled=int(w.sum()),p_low=low,p_high=high,
                           difference_pp=100*(high-low),score_max=fit['score_max'],rank_drops=len(fit['dropped']))
                if len(job['tested'])==1:row['OR']=float(np.exp(fit['beta'][job['tested'][0]]))
            except Exception as exc:row['error']=repr(exc)
            rows.append(row)
    return rows,correlations

def null_chunk(reps):
    job=WORK['jobs'][('C3','primary','Both_up')]
    p=job['payload'];null=job['null'];rows=[]
    probability=expit(p['X']@null['beta'])
    for rep in reps:
        rng=np.random.default_rng(np.random.SeedSequence([SEEDS['C3'],rep,150]))
        y=rng.binomial(1,probability).astype(float)
        row=dict(replicate=rep,valid=False)
        try:
            fit=fit_firth(p['X'],y,start=null['beta'],protected=p['protected'])
            lr,pv,restricted=test_firth(fit,job['tested'])
            row.update(valid=True,LR=lr,p_asymptotic=pv,score_max=fit['score_max'],null_score_max=restricted['score_max'])
        except Exception as exc:row['error']=repr(exc)
        rows.append(row)
    return rows

def run(args):
    cache=Path(args.repo)/'.work'
    with (cache/'association_fits.pkl').open('rb') as handle:jobs=pickle.load(handle)
    frame,*_=load_analysis(args.data)
    records=[];rho=[];nulls=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
                             initargs=(jobs,frame.CAU.to_numpy(),frame.AOR.to_numpy())) as pool:
        tasks=[pool.submit(bootstrap_chunk,list(range(i,min(i+25,args.bootstrap)))) for i in range(0,args.bootstrap,25)]
        for i,future in enumerate(as_completed(tasks),1):
            rows,cor=future.result();records.extend(rows);rho.extend(cor)
            if i%10==0:print('Respondent-bootstrap chunks:',i,'/',len(tasks),flush=True)
        tasks=[pool.submit(null_chunk,list(range(i,min(i+25,args.null_draws)))) for i in range(0,args.null_draws,25)]
        for future in as_completed(tasks):nulls.extend(future.result())
    draws=pd.DataFrame(records).sort_values(['group','scenario','outcome','replicate'])
    correlation=pd.DataFrame(rho).sort_values('replicate')
    null=pd.DataFrame(nulls).sort_values('replicate')
    summaries=[]
    for (group,scenario,outcome),part in draws.groupby(['group','scenario','outcome'],sort=False):
        good=part.loc[part.valid]
        row=dict(group=group,scenario=scenario,outcome=outcome,requested=len(part),valid=len(good),
                 max_score=float(good.score_max.max()),rank_drop_replicates=int(good.rank_drops.gt(0).sum()))
        for field in ['p_low','p_high','difference_pp','OR']:
            values=good[field].dropna()
            label='OR_boot' if field=='OR' else field
            if len(values):row[label+'_CI95_low'],row[label+'_CI95_high']=values.quantile([.025,.975])
        summaries.append(row)
    summary=pd.DataFrame(summaries)
    save_table(args.output,PHASE,'bootstrap_intervals',summary)
    estimate=stats.spearmanr(frame.CAU,frame.AOR)
    lo,hi=correlation.rho.quantile([.025,.975])
    save_table(args.output,PHASE,'cau_aor_correlation',pd.DataFrame([dict(n=323,rho=float(estimate.statistic),p_raw=float(estimate.pvalue),
        CI95_low=lo,CI95_high=hi,bootstrap_valid=len(correlation),seed=SEEDS['Components'])]))
    job=jobs[('C3','primary','Both_up')]
    observed=2*(job['fit']['pll']-job['null']['pll'])
    good=null.loc[null.valid]
    exceed=int(good.LR.ge(observed-1e-12).sum())
    pmc=(exceed+1)/(len(null)+1) if len(good)==len(null) else np.nan
    save_table(args.output,PHASE,'c3_parametric_null',pd.DataFrame([dict(requested=len(null),valid=len(good),observed_LR=observed,
        p_MC=pmc,MC_SE=float(np.sqrt(pmc*(1-pmc)/(len(null)+1))),
        rejection_fraction=float(good.p_asymptotic.lt(.05).mean()),seed=SEEDS['C3'])]))
    # Detailed resamples stay local; public outputs contain aggregate results.
    draws.to_csv(cache/'association_bootstrap_draws.csv',index=False)
    correlation.to_csv(cache/'correlation_bootstrap_draws.csv',index=False)
    null.to_csv(cache/'c3_null_draws.csv',index=False)
    write_json(Path(args.output)/'resampling_specification.json',dict(seeds=SEEDS,seed_tags=TAGS,bootstrap=args.bootstrap,
        null_draws=args.null_draws,unit='Paired respondents',scales='Fixed full-cohort SDs and spline knots',
        failures='No replacement draws; failures are retained and checked.'))
    if not draws.valid.all() or not null.valid.all():raise RuntimeError('Some resampling fits failed; inspect local diagnostics.')
    finish_phase(args.output,PHASE)
    return summary
