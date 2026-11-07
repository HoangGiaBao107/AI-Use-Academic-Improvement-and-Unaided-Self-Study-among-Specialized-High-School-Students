"""Cohort accounting, measurement checks, and report descriptives."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from .data import load_analysis,BLOCKS,bh_adjust
from .storage import save_table,finish_phase,write_json

def quality_flags(path,raw):
    text=raw.replace('',pd.NA)
    items=[c for group in BLOCKS.values() for c in group]
    lik=text[items].apply(lambda c:c.str.extract(r'^([1-5])',expand=False).astype(float))
    def longest(a):
        best=run=0;last=np.nan
        for value in a:
            run=run+1 if np.isfinite(value) and value==last else (1 if np.isfinite(value) else 0)
            best=max(best,run);last=value
        return best
    flat=pd.DataFrame({b:lik[v].notna().all(axis=1)&lik[v].std(axis=1,ddof=0).eq(0) for b,v in BLOCKS.items()}).sum(axis=1)
    straight=(lik.notna().all(axis=1)&lik.std(axis=1,ddof=0).eq(0))|lik.apply(lambda x:longest(x.to_numpy(float)),axis=1).ge(15)|flat.ge(5)
    encoded=np.column_stack([pd.factorize(text[c],sort=True)[0] for c in text])
    near=np.zeros(len(text),bool)
    for i in range(len(text)):
        valid=(encoded>=0)&(encoded[i]>=0);den=valid.sum(axis=1)
        sim=np.divide(((encoded==encoded[i])&valid).sum(axis=1),den,out=np.full(len(text),np.nan),where=den>=np.ceil(.9*encoded.shape[1]))
        sim[i]=np.nan;near[i]=np.nanmax(sim)>=.9
    keys=lik.fillna(-1).astype(int).astype(str).agg('|'.join,axis=1)
    demo=text[['A1','A2','A3']].fillna('(Missing)').agg('|'.join,axis=1)
    purposes=text.B5.map(lambda v:len(set(t.strip() for t in __import__('re').split(r'[;,]',str(v)) if t.strip())) if pd.notna(v) else np.nan)
    q=pd.DataFrame(dict(straightline=straight,exact_raw=pd.read_csv(path,dtype='string',keep_default_na=False).duplicated(keep=False),
        exact_normalized=text.duplicated(keep=False),near_duplicate=near,duplicate_likert=keys.duplicated(keep=False),
        duplicate_likert_demo=(keys+'#'+demo).duplicated(keep=False),B5_over_limit=purposes.gt(5)))
    q['any_review']=q.any(axis=1)
    retained=text[BLOCKS['CAU']].notna().all(axis=1)
    return q.loc[retained].reset_index(drop=True)

def prepare(args):
    f,t,raw,headers,sha=load_analysis(args.data);phase='01_data'
    quality=quality_flags(args.data,raw)
    assert quality.straightline.sum()==52 and quality.any_review.sum()==122
    assert np.array_equal(quality.straightline,f.straightline) and np.array_equal(quality.any_review,f.any_review)
    save_table(args.output,phase,'participant_flow',pd.DataFrame([
        ['Invited',522,np.nan],['Responded',358,100*358/522],['Did not respond',164,np.nan],
        ['Not in final analysis',35,np.nan],['English export',len(raw),np.nan],['Analyzed',len(f),100*len(f)/358]],
        columns=['stage','n','percent_of_relevant_denominator']))
    save_table(args.output,phase,'quality_flags',pd.DataFrame([dict(flag=c,n=int(quality[c].sum()),percent=100*quality[c].mean()) for c in quality]))
    save_table(args.output,phase,'export_missingness',pd.DataFrame([dict(variable=c,missing=int(raw[c].eq('').sum())) for c in raw]))
    save_table(args.output,phase,'english_item_dictionary',pd.DataFrame([dict(variable=c,wording=header,
        role='Excluded from models for content overlap' if c=='N2' else 'English translation of administered Vietnamese item') for c,header in headers.items()]))
    write_json(Path(args.output)/'data_manifest.json',dict(input_name=Path(args.data).name,input_sha256=sha,
        export_rows=len(raw),analysis_rows=len(f),columns=len(raw.columns),
        upstream_screening='Qualitative exclusions before export cannot be replayed: excluded records and log were not retained.',
        recruitment='Google Forms, 28-30 September 2026; one specialized school; competition-team eligibility; class monitors.',
        N3='Change in ability to study without help, not study duration and not specific to AI withdrawal.',
        A7='Previous school year; not an established pre-AI baseline.'))
    cache=Path(args.repo)/'.work';cache.mkdir(exist_ok=True)
    # Row-level intermediate data are local-only and excluded from Git.
    f.to_pickle(cache/'analysis_frame.pkl')
    finish_phase(args.output,phase)
    return f

def alpha(a):
    k=a.shape[1];total=a.sum(axis=1).var(ddof=1)
    return float(k/(k-1)*(1-a.var(axis=0,ddof=1).sum()/total)) if total>0 else np.nan

def describe(args):
    f,t,raw,headers,sha=load_analysis(args.data);phase='02_measurement'
    rows=[]
    for key in ['A1','A2','A3','A5','A7']:
        for value,n in t[key].value_counts().items():rows.append(dict(variable=key,category=value,n=int(n),percent=100*n/len(f)))
    save_table(args.output,phase,'sample_distributions',pd.DataFrame(rows))
    state=(2*f.N1_increase+f.N3_increase).astype(int)
    names={0:'Neither improved',1:'N3 only',2:'N1 only',3:'Both improved'}
    save_table(args.output,phase,'joint_improvement',pd.DataFrame([dict(state=names[k],n=int(state.eq(k).sum()),percent=100*state.eq(k).mean()) for k in range(4)]))
    names={0:'Neither declined',1:'Academic non-decline; self-study decline',2:'Academic decline; self-study non-decline',3:'Both declined'}
    save_table(args.output,phase,'joint_decline',pd.DataFrame([dict(state=names[k],n=int(f.state.eq(k).sum()),percent=100*f.state.eq(k).mean()) for k in range(4)]))
    outcome_rows=[]
    for target in ['N1','N3']:
        for label,n in f[target+'_3'].value_counts(sort=False).items():outcome_rows.append(dict(outcome=target,category=label,n=int(n),percent=100*n/len(f)))
    save_table(args.output,phase,'outcome_distributions',pd.DataFrame(outcome_rows))
    blocks={**BLOCKS,'RAI_peer3':['RAI1','RAI2','RAI3'],'AOR_no2':['AOR1','AOR3','AOR4','AOR5'],'AOR_no3':['AOR1','AOR2','AOR4','AOR5']}
    rng=np.random.default_rng(20261025);reliability=[]
    for name,columns in blocks.items():
        a=f[columns].to_numpy(float)
        values=[alpha(a[rng.integers(0,len(a),len(a))]) for _ in range(args.bootstrap)]
        lo,hi=np.nanquantile(values,[.025,.975]);score=a.mean(axis=1)
        reliability.append(dict(scale=name,n=len(a),items=len(columns),mean=float(score.mean()),SD=float(score.std(ddof=1)),
                                alpha=alpha(a),alpha_CI95_low=lo,alpha_CI95_high=hi,bootstrap=args.bootstrap,seed=20261025))
    save_table(args.output,phase,'scale_reliability',pd.DataFrame(reliability))
    correlations=[]
    for name in list(blocks)+['RAI4','RAI5']:
        for outcome in ['N1','N3']:
            result=stats.spearmanr(f[name],f[outcome])
            correlations.append(dict(variable=name,outcome=outcome,n=len(f),rho=float(result.statistic),p_raw=float(result.pvalue)))
    cor=pd.DataFrame(correlations);assert len(cor)==24
    cor['q_BH_24']=bh_adjust(cor.p_raw);save_table(args.output,phase,'descriptive_correlations',cor)
    sparse=f.assign(group=t.A7.replace({'Fair':'Lower','Pass':'Lower','Not passed':'Lower',"Don't remember":'Unknown'})).groupby(['A1_label','A3_label','group']).Both_up.agg(['size','sum']).reset_index()
    sparse=sparse.rename(columns={'size':'n','sum':'joint_improvements'})
    save_table(args.output,phase,'sparse_covariate_cells',sparse)
    pairs=f.groupby(['CAU','AOR']).size().rename('n').reset_index()
    save_table(args.output,phase,'cau_aor_score_pairs',pairs)
    context=[]
    for block in ['TAG','AIC','RAI']:
        for item in BLOCKS[block]:
            value=f[item];context.append(dict(item=item,n=len(f),agree_n=int(value.ge(4).sum()),agree_percent=100*value.ge(4).mean(),
                neutral_n=int(value.eq(3).sum()),neutral_percent=100*value.eq(3).mean(),disagree_n=int(value.le(2).sum()),disagree_percent=100*value.le(2).mean(),wording=headers[item]))
    save_table(args.output,phase,'school_classroom_items',pd.DataFrame(context))
    finish_phase(args.output,phase)
    return pd.DataFrame(reliability)
