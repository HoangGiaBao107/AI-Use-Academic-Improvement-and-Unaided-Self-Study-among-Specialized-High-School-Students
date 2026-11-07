"""Recompute report associations and their declared BH-FDR scopes."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.special import expit
from .data import load_analysis,reference_scales,bh_adjust
from .designs import SCENARIOS,CAU_SCENARIOS,OUTCOMES,make_aor_design,make_cau_design
from .firth import fit_firth,test_firth,profile_ci
from .storage import save_table,finish_phase,write_json

PHASE='03_associations'

def fit_record(payload,y,tested,group,scenario,target,low_matrix,high_matrix):
    fit=fit_firth(payload['X'],y,protected=payload['protected'])
    lr,p,null=test_firth(fit,tested)
    rec=dict(group=group,scenario=scenario,outcome=target,n=len(y),events=int(y.sum()),
             df=len(tested),LR=lr,p_raw=p,score_max=fit['score_max'],null_score_max=null['score_max'],
             p_low=float(expit(low_matrix@fit['beta']).mean()),p_high=float(expit(high_matrix@fit['beta']).mean()))
    rec['difference_pp']=100*(rec['p_high']-rec['p_low'])
    if len(tested)==1:
        ci,residual=profile_ci(fit,tested[0])
        rec.update(OR_per_SD=float(np.exp(fit['beta'][tested[0]])),OR_CI95_low=float(np.exp(ci[0])),
                   OR_CI95_high=float(np.exp(ci[1])),profile_residual=residual)
    return rec,fit,null

def fit_all(data_path):
    frame,_,_,_,digest=load_analysis(data_path);scales=reference_scales(frame)
    records=[];jobs={};coefficients=[]
    for scenario in CAU_SCENARIOS:
        p=make_cau_design(frame,scenario,scales)
        for k,outcome in enumerate(OUTCOMES):
            r,f,null=fit_record(p,p['Y'][:,k],p['tested'],'CAU',scenario,outcome,p['X3'],p['X4'])
            records.append(r)
            if scenario=='primary' and outcome in ['Both_up','N1_up']:
                jobs[('CAU',scenario,outcome)]=dict(payload=p,fit=f,null=null,y=p['Y'][:,k],tested=p['tested'],Xlow=p['X3'],Xhigh=p['X4'],indices=np.arange(len(frame)))
        print('CAU:',scenario,flush=True)
    for scenario in SCENARIOS:
        p=make_aor_design(frame,scenario,scales)
        for target in ['P_all','P_conditional','Both_up']:
            mask=p['eligible'] if target=='P_conditional' else np.ones(len(p['X']),bool)
            y=(p['n3'] if target=='P_conditional' else p['focal']) if target!='Both_up' else frame.loc[p['indices'],'Both_up'].to_numpy(float)
            pp={**p,'X':p['X'][mask]}
            low,high=p['X2'][mask],p['X3'][mask]
            r,f,null=fit_record(pp,y[mask],p['tested'],'C3' if target=='Both_up' else 'C2',scenario,target,low,high)
            records.append(r)
            if target=='Both_up' or scenario=='primary':
                jobs[(r['group'],scenario,target)]=dict(payload=pp,fit=f,null=null,y=y[mask],tested=p['tested'],Xlow=low,Xhigh=high,indices=p['indices'][mask])
            if target=='Both_up':
                coefficients.extend(dict(scenario=scenario,term=t,beta=float(b)) for t,b in zip(p['names'],f['beta']))
        print('AOR:',scenario,flush=True)
    p=make_aor_design(frame,'primary',scales)
    for target in ['N1_up','N3_up']:
        y=frame[target].to_numpy(float)
        r,f,null=fit_record(p,y,p['tested'],'Components','primary',target,p['X2'],p['X3'])
        records.append(r)
        jobs[('Components','primary',target)]=dict(payload=p,fit=f,null=null,y=y,tested=p['tested'],Xlow=p['X2'],Xhigh=p['X3'],indices=np.arange(len(frame)))
    return pd.DataFrame(records),jobs,pd.DataFrame(coefficients),scales,digest

def run(args):
    tests,jobs,coefficients,scales,digest=fit_all(args.data)
    provenance=Path(args.repo)/'provenance/historical_tests.csv'
    frozen=pd.read_csv(provenance)
    old=frozen.loc[frozen.family.eq('old44')].copy();rai=frozen.loc[frozen.family.eq('RAI4_40')].copy()
    assert len(old)==44 and len(rai)==40
    cau=tests.loc[tests.group.eq('CAU')].copy()
    c2=tests.loc[tests.group.eq('C2')].copy()
    c3=tests.loc[tests.group.eq('C3')].copy()
    comp=tests.loc[tests.group.eq('Components')].copy()
    cau['q_BH_80']=bh_adjust(np.r_[rai.p_raw,cau.p_raw])[40:]
    c2['q_BH_20']=bh_adjust(c2.p_raw)
    c3['q_BH_10']=bh_adjust(c3.p_raw)
    comp['q_BH_2']=bh_adjust(comp.p_raw)
    scope=pd.concat([old.assign(group='Historical'),c2,rai.assign(group='Historical'),cau,c3,comp],ignore_index=True)
    assert len(scope)==156
    scope['q_BH_144']=np.r_[bh_adjust(scope.p_raw.iloc[:144]),np.full(12,np.nan)]
    scope['q_BH_154']=np.r_[bh_adjust(scope.p_raw.iloc[:154]),np.full(2,np.nan)]
    scope['q_BH_156']=bh_adjust(scope.p_raw)
    for tab,offset in [(c2,44),(cau,104),(c3,144),(comp,154)]:
        for col in ['q_BH_144','q_BH_154','q_BH_156']:
            tab[col]=scope[col].iloc[offset:offset+len(tab)].to_numpy()
    for name,tab in [('cau_sensitivity',cau),('c2_sensitivity',c2),('c3_sensitivity',c3),('c3_components',comp),('multiplicity_registry',scope),('c3_coefficients',coefficients)]:
        save_table(args.output,PHASE,name,tab)
    main=pd.concat([
        cau.loc[cau.scenario.eq('primary')&cau.outcome.eq('N1_up')].assign(question='C1a',adjusted_p=lambda z:z.q_BH_80),
        c3.loc[c3.scenario.eq('primary')].assign(question='C3',adjusted_p=lambda z:z.q_BH_10),
        cau.loc[cau.scenario.eq('primary')&cau.outcome.eq('Both_up')].assign(question='C1b',adjusted_p=lambda z:z.q_BH_80),
        c2.loc[c2.scenario.eq('primary')].assign(question='C2',adjusted_p=lambda z:z.q_BH_20)],ignore_index=True)
    save_table(args.output,PHASE,'main_models',main)
    cache=Path(args.repo)/'.work';cache.mkdir(exist_ok=True)
    with (cache/'association_fits.pkl').open('wb') as handle:pickle.dump(jobs,handle)
    write_json(Path(args.output)/'association_specification.json',dict(input_sha256=digest,scales=scales,
        frozen_test_count=len(frozen),recomputed_test_count=len(tests),history='Frozen out-of-scope p-values are used only for multiplicity provenance.'))
    finish_phase(args.output,PHASE)
    return main
