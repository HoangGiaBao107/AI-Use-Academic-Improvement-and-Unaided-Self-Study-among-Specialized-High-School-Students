"""Continuous CAU-by-AOR interaction and school-context descriptives."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from threadpoolctl import threadpool_limits
from .data import load_analysis,reference_scales,bh_adjust,BLOCKS
from .designs import make_aor_design
from .firth import fit_firth,test_firth,profile_ci
from .storage import save_table,finish_phase

PHASE='06_interaction_context'
SEED=20261024
SPECS=[('N3_primary','N3_up','primary'),('N1_secondary','N1_up','primary'),
       ('N3_AOR_no2','N3_up','AOR_no2'),('N3_AOR_no3','N3_up','AOR_no3'),
       ('N3_no_straightline','N3_up','no_straightline'),('N3_no_review_flags','N3_up','no_review_flags'),
       ('N3_B2_categorical','N3_up','B2_categorical'),('N3_expanded','N3_up','expanded_constructs'),
       ('N3_expanded_AOR_no3','N3_up','expanded_no3'),('N3_nonlinear_mains','N3_up','nonlinear_scores')]
WORK={}

def initialize(X,y,start,null,boot_indices,null_outcomes):
    threadpool_limits(1)
    WORK.update(X=X,y=y,start=start,null=null,boot=boot_indices,null_outcomes=null_outcomes)

def resample_task(task):
    kind,reps=task;X=WORK['X'];j=X.shape[1]-1;rows=[]
    for rep in reps:
        row=dict(kind=kind,replicate=rep,valid=False)
        try:
            if kind=='case':
                w=np.bincount(WORK['boot'][rep],minlength=len(X)).astype(float)
                fit=fit_firth(X,WORK['y'],weights=w,start=WORK['start'],protected=[j])
                row.update(valid=True,OR=float(np.exp(fit['beta'][j])),score_max=fit['score_max'])
            else:
                y=WORK['null_outcomes'][rep]
                fit=fit_firth(X,y,start=WORK['null']['beta'],protected=[j])
                lr,p,null=test_firth(fit,[j]);row.update(valid=True,LR=lr,score_max=fit['score_max'],null_score_max=null['score_max'])
        except Exception as exc:row['error']=repr(exc)
        rows.append(row)
    return rows

def run(args):
    frame,_,_,headers,_=load_analysis(args.data);scales=reference_scales(frame)
    rows=[];primary=None
    for name,outcome,scenario in SPECS:
        f=frame.copy();ss=dict(scales)
        if scenario=='expanded_no3':
            scenario='expanded_constructs';f['AOR']=f.AOR_no3;ss['AOR']=scales['AOR_no3']
        p=make_aor_design(f,scenario,ss)
        X=p['X'];names=p['names'];a=names.index('CAU_z');b=names.index('AOR_z')
        X=np.column_stack([X,X[:,a]*X[:,b]]);j=X.shape[1]-1
        y=frame.loc[p['indices'],outcome].to_numpy(float)
        fit=fit_firth(X,y,protected=p['protected']+[j]);lr,pv,null=test_firth(fit,[j]);ci,residual=profile_ci(fit,j)
        rows.append(dict(specification=name,outcome=outcome,n=len(y),interaction_OR=float(np.exp(fit['beta'][j])),
            OR_CI95_low=float(np.exp(ci[0])),OR_CI95_high=float(np.exp(ci[1])),p_raw=pv,LR=lr,
            score_max=fit['score_max'],null_score_max=null['score_max'],profile_residual=residual))
        if name=='N3_primary':primary=(X,y,fit,null,lr)
    tests=pd.DataFrame(rows);tests['q_BH_10']=bh_adjust(tests.p_raw)
    save_table(args.output,PHASE,'continuous_interaction_tests',tests)
    X,y,fit,null,observed=primary
    rng=np.random.default_rng(SEED+1000);indices=rng.integers(0,len(y),size=(args.bootstrap,len(y)))
    rng=np.random.default_rng(SEED+9001);outcomes=np.stack([rng.binomial(1,expit(X@null['beta'])).astype(float) for _ in range(args.null_draws)])
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
                             initargs=(X,y,fit['beta'],null,indices,outcomes)) as pool:
        tasks=[('case',list(range(i,min(i+25,args.bootstrap)))) for i in range(0,args.bootstrap,25)]
        tasks += [('null',list(range(i,min(i+25,args.null_draws)))) for i in range(0,args.null_draws,25)]
        for future in as_completed([pool.submit(resample_task,t) for t in tasks]):records.extend(future.result())
    records=pd.DataFrame(records).sort_values(['kind','replicate'])
    boot=records.loc[records.kind.eq('case')&records.valid];nulls=records.loc[records.kind.eq('null')&records.valid]
    lo,hi=boot.OR.quantile([.025,.975]);exceed=int(nulls.LR.ge(observed-1e-12).sum())
    summary=pd.DataFrame([dict(case_requested=args.bootstrap,case_valid=len(boot),OR_boot_CI95_low=lo,OR_boot_CI95_high=hi,
        null_requested=args.null_draws,null_valid=len(nulls),p_MC=(exceed+1)/(args.null_draws+1),seed=SEED)])
    save_table(args.output,PHASE,'interaction_resampling',summary)
    records.to_csv(Path(args.repo)/'.work/interaction_resamples.csv',index=False)
    context=[]
    for block in ['TAG','AIC','RAI']:
        for item in BLOCKS[block]:
            a=frame[item];context.append(dict(item=item,n=len(a),agree_n=int(a.ge(4).sum()),agree_percent=100*a.ge(4).mean(),
                neutral_n=int(a.eq(3).sum()),neutral_percent=100*a.eq(3).mean(),disagree_n=int(a.le(2).sum()),disagree_percent=100*a.le(2).mean(),wording=headers[item]))
    save_table(args.output,PHASE,'school_classroom_context',pd.DataFrame(context))
    if not records.valid.all():raise RuntimeError('Interaction resampling failures are retained; inspect local diagnostics.')
    finish_phase(args.output,PHASE)
    return tests
