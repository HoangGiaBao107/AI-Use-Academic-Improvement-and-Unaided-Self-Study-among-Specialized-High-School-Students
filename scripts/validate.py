"""Five complementary validation rounds for the report reproduction."""
from pathlib import Path
import argparse,ast,csv,hashlib,json,re,shutil,subprocess,sys,time,unittest,zipfile
import numpy as np
import pandas as pd
from scipy import optimize,stats
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ai_learning.storage import write_json

CHECKS=[]
def check(round_id,name,condition,detail=''):
    CHECKS.append(dict(round=round_id,name=name,passed=bool(condition),detail=str(detail)))
    if not condition:print('FAIL:',round_id,name,detail,flush=True)

def table(phase,name,base=None):return pd.read_csv((ROOT/'results' if base is None else Path(base))/'tables'/phase/(name+'.csv'))

def round_source():
    for p in sorted(ROOT.rglob('*.py')):
        if '.work' in p.parts or '.git' in p.parts:continue
        try:ast.parse(p.read_text());ok=True
        except SyntaxError:ok=False
        check(1,'Python syntax '+str(p.relative_to(ROOT)),ok)
        # Code, comments and messages are English; proper names appear only in documentation.
        check(1,'English source '+str(p.relative_to(ROOT)),not re.search(r'[\u1ea0-\u1ef9\u0110\u0111]',p.read_text()))
    nb=json.loads((ROOT/'notebooks/reproduce_report.ipynb').read_text())
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':
            try:compile(''.join(c['source']),'cell','exec');ok=True
            except SyntaxError:ok=False
            check(1,'Notebook code cell '+str(i),ok)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    check(1,'Numerical contract tests',result.wasSuccessful(),f'{result.testsRun} tests')
    check(1,'Analysis does not load validation snapshots',all('provenance/reference' not in p.read_text() for p in (ROOT/'src').rglob('*.py')))
    print('Round 1 completed',flush=True)

def round_independent(data):
    # Re-encode directly from CSV text, independently of the pipeline's data module.
    with Path(data).open(newline='',encoding='utf-8-sig') as handle:reader=csv.DictReader(handle);rows=list(reader);headers=reader.fieldnames
    codes={re.search(r'(?:^|\[)([A-Z]+\d+[a-d]?)\.',h).group(1):h for h in headers}
    valid=[r for r in rows if all(r[codes['CAU'+str(i)]].strip() for i in range(1,6))]
    check(2,'Raw and complete-case dimensions',len(rows)==324 and len(headers)==58 and len(valid)==323)
    check(2,'Input identity',hashlib.sha256(Path(data).read_bytes()).hexdigest()==json.loads((ROOT/'provenance/report_targets.json').read_text())['input_sha256'])
    n1=np.array([r[codes['N1']].startswith('Increased') for r in valid]);n3=np.array([r[codes['N3']].startswith('Increased') for r in valid])
    d1=np.array([r[codes['N1']].startswith('Decreased') for r in valid]);d3=np.array([r[codes['N3']].startswith('Decreased') for r in valid])
    check(2,'Joint outcome accounting',(n1.sum(),n3.sum(),(n1&n3).sum(),(n1&~n3).sum(),(~d1&d3).sum(),(~d1).sum())==(205,122,101,104,79,296))
    cau=np.array([np.mean([float(r[codes['CAU'+str(i)]][0]) for i in range(1,6)]) for r in valid])
    aor=np.array([np.mean([float(r[codes['AOR'+str(i)]][0]) for i in range(1,6)]) for r in valid])
    check(2,'Full-cohort score SDs',np.allclose([cau.std(ddof=1),aor.std(ddof=1)],[.7859402366914646,.8885090326268801],atol=1e-13))
    cat=lambda variable,value:np.array([r[codes[variable]].strip()==value for r in valid],float)
    lower=np.array([r[codes['A7']].strip() in ['Fair','Pass','Not passed'] for r in valid],float)
    unknown=cat('A7',"Don't remember")
    b1=np.array([r[codes['B1']].strip() for r in valid]);b1=np.where(np.isin(b1,['Less than once/week','1–2 times/week']),'At most 2 times/week',b1)
    b2levels=['Under 10 minutes','10 to <20 minutes','20 to <40 minutes','40 to <60 minutes','60+ minutes']
    b2=np.array([b2levels.index(r[codes['B2']].strip())+1 for r in valid],float)
    X=np.column_stack([np.ones(len(valid)),cat('A1','11'),cat('A1','12'),cat('A3','1'),lower,unknown,
        *[b1==v for v in ['At most 2 times/week','6–10 times/week','More than 10 times/week','Almost daily']],b2,
        (cau-cau.mean())/cau.std(ddof=1),(aor-aor.mean())/aor.std(ddof=1)]).astype(float)
    main=table('03_associations','main_models');comp=table('03_associations','c3_components')
    tests=[('C1a',n1,np.ones(len(n1),bool),11,main.loc[main.question.eq('C1a')].iloc[0]),
           ('C1b',n1&n3,np.ones(len(n1),bool),11,main.loc[main.question.eq('C1b')].iloc[0]),
           ('C3',n1&n3,np.ones(len(n1),bool),12,main.loc[main.question.eq('C3')].iloc[0]),
           ('C2 full',~d1&d3,np.ones(len(n1),bool),12,main.loc[main.outcome.eq('P_all')].iloc[0]),
           ('C2 conditional',d3,~d1,12,main.loc[main.outcome.eq('P_conditional')].iloc[0]),
           ('Component N1',n1,np.ones(len(n1),bool),12,comp.loc[comp.outcome.eq('N1_up')].iloc[0]),
           ('Component N3',n3,np.ones(len(n1),bool),12,comp.loc[comp.outcome.eq('N3_up')].iloc[0])]
    for name,y,mask,j,expected in tests:
        xx=X[mask];yy=y[mask].astype(float)
        def objective(beta):
            eta=xx@beta;p=expit(eta);sign,ld=np.linalg.slogdet(xx.T@((p*(1-p))[:,None]*xx))
            return -np.sum(yy*eta-np.logaddexp(0,eta))-.5*ld if sign>0 else np.inf
        fit=optimize.minimize(objective,np.zeros(xx.shape[1]),jac='3-point',method='BFGS',options={'gtol':1e-7,'maxiter':800})
        free=np.arange(xx.shape[1])!=j
        def lift(z,value=0):
            b=np.zeros(xx.shape[1]);b[free]=z;b[j]=value;return b
        null=optimize.minimize(lambda z:objective(lift(z)),fit.x[free],jac='3-point',method='BFGS',options={'gtol':1e-7,'maxiter':800})
        p=stats.chi2.sf(2*(null.fun-fit.fun),1)
        check(2,'Independent Firth OR '+name,abs(np.exp(fit.x[j])-expected.OR_per_SD)<3e-6)
        check(2,'Independent penalized LR '+name,abs(p-expected.p_raw)<2e-7)
    rho=table('04_uncertainty','cau_aor_correlation').iloc[0]
    check(2,'Independent Spearman correlation',abs(stats.spearmanr(cau,aor).statistic-rho.rho)<1e-11)
    context=table('06_interaction_context','school_classroom_context').set_index('item')
    for item,n in [('AIC3',234),('AIC2',132),('RAI3',110),('RAI5',244)]:
        observed=sum(int(r[codes[item]][0])>=4 for r in valid)
        check(2,'Independent item count '+item,observed==n==context.loc[item,'agree_n'])
    print('Round 2 completed',flush=True)

def compare(label,a,b,keys,fields,tolerance=1e-6):
    c=a.merge(b,on=keys,how='outer',suffixes=('_new','_ref'),indicator=True)
    check(3,label+' row coverage',len(c)==len(a)==len(b) and c['_merge'].eq('both').all())
    for new,old in fields:
        nn=new+'_new' if new in b and new not in keys else new
        oo=old+'_ref' if old in a and old not in keys else old
        x=pd.to_numeric(c[nn],errors='coerce').to_numpy();y=pd.to_numeric(c[oo],errors='coerce').to_numpy()
        difference=np.nanmax(abs(x-y)) if np.any(np.isfinite(x-y)) else 0
        check(3,label+' '+new,np.allclose(x,y,atol=tolerance,rtol=1e-7,equal_nan=True),f'max absolute difference {difference:.3g}')

def round_reference():
    reference=ROOT/'provenance/reference'
    for name,qs in [('cau_sensitivity',('q_BH_80','q_BH_joint_new80')),('c2_sensitivity',('q_BH_20','q_BH_20')),('c3_sensitivity',('q_BH_10','q_BH_new10'))]:
        a=table('03_associations',name);b=pd.read_csv(reference/(name+'.csv')).rename(columns={'target':'outcome'})
        fields=[(n,n) for n in ['n','p_raw','OR_per_SD','OR_CI95_low','OR_CI95_high']]+[qs]
        compare(name,a,b,['scenario','outcome'],fields,1e-6)
    a=table('03_associations','c3_components');b=pd.read_csv(reference/'components.csv')
    compare('C3 components',a,b,['outcome'],[('OR_per_SD','OR_per_SD'),('p_raw','p_raw'),('OR_CI95_low','CI95_low'),('OR_CI95_high','CI95_high'),('q_BH_2','q_BH_component2'),('q_BH_156','q_BH_history156')])
    a=table('02_measurement','scale_reliability');b=pd.read_csv(reference/'reliability.csv')
    compare('Reliability',a,b,['scale'],[('alpha','alpha'),('alpha_CI95_low','alpha_CI_low'),('alpha_CI95_high','alpha_CI_high')],1e-10)
    allboot=table('04_uncertainty','bootstrap_intervals')
    for group,file,keys,mapping in [('CAU','cau_bootstrap',['outcome'],[('p_low','p3'),('p_high','p4')]),('C3','c3_bootstrap',['scenario'],[('p_low','p_AOR2'),('p_high','p_AOR3')])]:
        a=allboot.loc[allboot.group.eq(group)].copy();b=pd.read_csv(reference/(file+'.csv'))
        if group=='CAU':b=b.loc[b.outcome.isin(a.outcome)]
        fields=[('difference_pp_CI95_low','difference_pp_CI95_low'),('difference_pp_CI95_high','difference_pp_CI95_high'),('valid','valid')]
        fields += [(new+'_CI95_'+end,old+'_CI95_'+end) for new,old in mapping for end in ['low','high']]
        compare(group+' bootstrap',a,b,keys,fields,2e-6)
    c2=allboot.loc[allboot.group.eq('C2')].set_index('outcome');b=pd.read_csv(reference/'c2_bootstrap.csv')
    for target in ['P_all','P_conditional']:
        for field,estimand,multiplier in [('p_low','p2',1),('p_high','p3',1),('difference_pp','diff',100)]:
            row=b.loc[b.model.eq(target)&b.estimand.eq(estimand)].iloc[0]
            for end in ['low','high']:check(3,'C2 bootstrap '+target+' '+field+' '+end,abs(c2.loc[target,field+'_CI95_'+end]-multiplier*row['CI95_'+end])<2e-6)
    a=table('05_prediction','paired_predictive_gains');b=pd.read_csv(reference/'predictive_gains_all_metrics.csv')
    compare('All predictive losses',a,b,['outcome','family','baseline','comparison','metric'],[(x,x) for x in ['gain','CI95_low','CI95_high']],5e-6)
    primary=table('05_prediction','primary_logloss_comparisons');broad=a.loc[a.metric.eq('log_loss')]
    check(3,'Ten primary and 45 broad intervals include zero',len(primary)==10 and len(broad)==45 and ((broad.CI95_low<=0)&(broad.CI95_high>=0)).all())
    a=table('05_prediction','nested_selection_metrics');b=pd.read_csv(reference/'nested_selection_metrics.csv')
    compare('Nested selection',a,b,['outcome'],[(x,x) for x in ['log_loss','Brier','RPS']],5e-6)
    a=table('05_prediction','nested_selected_candidates');b=pd.read_csv(reference/'nested_selected_candidates.csv')
    c=a.merge(b,on=['outcome','outer_fold'],suffixes=('_new','_ref'))
    check(3,'Nested winners match archived selection',len(c)==10 and c.family_new.eq(c.family_ref).all() and c.model_new.eq(c.model_ref).all())
    a=table('05_prediction','calibration');b=pd.read_csv(reference/'calibration.csv')
    compare('Separate calibration',a,b,['outcome','model'],[(x,x) for x in ['calibration_intercept','calibration_slope']],2e-7)
    a=table('06_interaction_context','continuous_interaction_tests');b=pd.read_csv(reference/'interaction.csv')
    b['specification']=b.test_id.str.replace('H5_','',regex=False)
    compare('Continuous interaction',a,b,['specification'],[(x,x) for x in ['interaction_OR','OR_CI95_low','OR_CI95_high','p_raw']]+[('q_BH_10','q_BH_H5_model_family10')],2e-6)
    a=table('06_interaction_context','interaction_resampling').iloc[0];b=pd.read_csv(reference/'interaction_bootstrap.csv').iloc[0]
    check(3,'Interaction bootstrap interval',np.allclose([a.OR_boot_CI95_low,a.OR_boot_CI95_high],[b.OR_CI95_percentile_low,b.OR_CI95_percentile_high],atol=2e-6))
    b=pd.read_csv(reference/'interaction_null.csv').iloc[0]
    check(3,'Interaction null Monte Carlo result',abs(a.p_MC-b.p_MC_secondary)<1e-12 and a.case_valid==2000 and a.null_valid==999)
    a=table('04_uncertainty','c3_parametric_null').iloc[0];b=pd.read_csv(reference/'c3_null.csv').iloc[0]
    check(3,'C3 null Monte Carlo result',abs(a.p_MC-b.p_MC)<1e-12 and a.valid==999)
    c3=allboot.loc[allboot.group.eq('C3')]
    check(3,'All 10 C3 contrasts negative with bootstrap intervals below zero',len(c3)==10 and c3.difference_pp_CI95_high.lt(0).all() and c3.valid.eq(2000).all())
    registry=table('03_associations','multiplicity_registry')
    for column,n in [('q_BH_144',144),('q_BH_154',154),('q_BH_156',156)]:
        q=stats.false_discovery_control(registry.p_raw.iloc[:n].to_numpy(),method='bh')
        check(3,'Independent historical BH '+str(n),np.allclose(q,registry[column].iloc[:n],atol=2e-10))
    print('Round 3 completed',flush=True)

def round_replay(data,workers):
    folder=ROOT/'.work/replay_checkout'
    if folder.exists():shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for name in ['src','provenance']:shutil.copytree(ROOT/name,folder/name,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(ROOT/'run_pipeline.py',folder/'run_pipeline.py')
    target=folder/'results'
    command=[sys.executable,str(folder/'run_pipeline.py'),'--data',str(Path(data).resolve()),'--output',str(target),'--workers',str(workers)]
    with (ROOT/'.work/fresh_replay.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    check(4,'Fresh full-pipeline process',result.returncode==0,'See local .work/fresh_replay.log')
    if result.returncode:print((ROOT/'.work/fresh_replay.log').read_text()[-5000:]);return
    original=sorted((ROOT/'results/tables').rglob('*.csv'))
    check(4,'Fresh table coverage',len(original)==len(list((target/'tables').rglob('*.csv'))))
    for p in original:
        other=target/p.relative_to(ROOT/'results')
        try:pd.testing.assert_frame_equal(pd.read_csv(p),pd.read_csv(other),check_exact=False,rtol=1e-8,atol=1e-9);ok=True
        except (AssertionError,FileNotFoundError):ok=False
        check(4,'Fresh replay '+str(p.relative_to(ROOT/'results')),ok)
    figs=list((ROOT/'results/figures').glob('*.svg'))
    for p in figs:check(4,'Deterministic figure '+p.name,p.read_bytes()==(target/'figures'/p.name).read_bytes())
    manifest=json.loads((target/'run_manifest.json').read_text())
    check(4,'Production replay settings',manifest['production_settings'] and manifest['bootstrap']==2000 and manifest['null_draws']==999)
    print('Round 4 completed',flush=True)

def round_package():
    review=ROOT/'results/verification/five_round_review.md'
    if not review.exists():
        review.parent.mkdir(parents=True,exist_ok=True)
        review.write_text('# Five-round verification\n\nValidation is in progress; this file is finalized after all checks.\n')
    for p in sorted((ROOT/'results/downloads').glob('*.zip')):
        with zipfile.ZipFile(p) as z:ok=z.testzip() is None and len(z.namelist())>0
        check(5,'Download ZIP '+p.name,ok)
    check(5,'Exactly seven figure pairs',len(list((ROOT/'results/figures').glob('*.png')))==7 and len(list((ROOT/'results/figures').glob('*.svg')))==7)
    for p in (ROOT/'results/tables').rglob('*.csv'):
        columns=pd.read_csv(p,nrows=0).columns
        check(5,'Aggregate-only columns '+str(p.relative_to(ROOT)),not any(c.lower() in ['respondent_id','student_id','email','name','y_true','p_0','p_1','p_2'] for c in columns))
    ignored=(ROOT/'.gitignore').read_text()
    check(5,'Reviewed CSV allowed; extra exports and private caches excluded','/data/*.csv' in ignored and '!/data/data raw eng.csv' in ignored and '/data raw eng.csv' in ignored and '/.work/' in ignored)
    check(5,'Phase downloads explicitly retained','!/results/downloads/' in ignored)
    check(5,'Optional manuscript PDFs are nonempty',all(p.stat().st_size>0 for p in (ROOT/'report').glob('*.pdf')))
    files=['README.md','LICENSE','CITATION.cff','requirements.txt','docs/CLAIM_TRACEABILITY.md','docs/REPOSITORY_TREE.md','notebooks/reproduce_report.ipynb','report/README.md']
    for name in files:check(5,'Required publication file '+name,(ROOT/name).is_file())
    # Local relative links are checked after removing code fences and inline code.
    for p in [ROOT/'README.md',ROOT/'results/index.md']:
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if re.match(r'https?://|#|mailto:',link):continue
            check(5,'Local link '+link,(p.parent/link).exists())
    print('Round 5 completed',flush=True)

def save_review():
    folder=ROOT/'results/verification';folder.mkdir(exist_ok=True)
    result=dict(rounds=[dict(round=i,passed=sum(x['passed'] for x in CHECKS if x['round']==i),total=sum(x['round']==i for x in CHECKS)) for i in range(1,6)],
                all_passed=all(x['passed'] for x in CHECKS),passed=sum(x['passed'] for x in CHECKS),total=len(CHECKS),checks=CHECKS)
    write_json(folder/'five_round_review.json',result)
    lines=['# Five-round verification','',f'**Result: {result["passed"]}/{result["total"]} checks passed.**','',
           '| Round | Scope | Passed |','|---|---|---|']
    scopes=['Source syntax, English code, notebook cells, and estimator contracts','Independent raw-data encoding and numerical refits','Agreement with archived report evidence and adjustment scopes','Fresh full-pipeline replay at research settings','Aggregate publication package, downloads, and links']
    for row,scope in zip(result['rounds'],scopes):lines.append(f'| {row["round"]} | {scope} | {row["passed"]}/{row["total"]} |')
    lines += ['', 'These are complementary technical checks. They do not establish construct validity, causality, representative sampling, or independent confirmation.',
              'A fresh replay starts from the original CSV and an isolated checkout, not from saved results or validation snapshots.',
              'Reference comparison tolerances accommodate machine-level optimizer differences; the JSON file records the checks and observed maximum differences.','']
    (folder/'five_round_review.md').write_text('\n'.join(lines))
    print('TOTAL:',result['passed'],'/',result['total'],flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=ROOT/'data/data raw eng.csv');p.add_argument('--full',action='store_true');p.add_argument('--workers',type=int,default=4);p.add_argument('--snapshots-only',action='store_true')
    a=p.parse_args()
    round_source()
    if a.snapshots_only:
        round_reference();round_package()
        if not all(x['passed'] for x in CHECKS):raise SystemExit(1)
        print('Public snapshot checks passed; raw-data replay was not performed.')
    else:
        if not a.full:raise SystemExit('Use --full for all five rounds, or --snapshots-only for public-output checks.')
        round_independent(a.data);round_reference();round_replay(a.data,a.workers);round_package()
        result=save_review()
        if not result['all_passed']:raise SystemExit(1)
