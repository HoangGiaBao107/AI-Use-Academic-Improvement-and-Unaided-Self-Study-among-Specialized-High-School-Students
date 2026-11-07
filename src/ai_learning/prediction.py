"""Repeated and nested cross-validation using the report's fixed candidates."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import pickle,warnings,time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from scipy.optimize import minimize,LinearConstraint
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.numdiff import approx_hess,approx_fprime
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .data import load_analysis,C_COLS
from .storage import save_table,finish_phase

SEED=20260101
RIDGE_C=1.0
FIT_MAXITER=600
OUTCOME_LEVELS={'N1':[1,2,3,4],'N3':[1,2,3,4,5]}
Q_COLS=[f'B1_dummy_{i}' for i in range(1,5)]+['B2_trend']
S=['CAU','AOR'];SP=S+['RAI4','RAI5'];L=['AIL'];E=['TAG','AIC','RAI_peer3']
SPECS={'M0':C_COLS,'MQ':C_COLS+Q_COLS,'MS':C_COLS+S,'MQS':C_COLS+Q_COLS+S,
 'MSplus':C_COLS+SP,'MQSplus':C_COLS+Q_COLS+SP,'MLE':C_COLS+L+E,
 'MLEQ':C_COLS+L+E+Q_COLS,'MLESplus':C_COLS+L+E+SP,'MLEQSplus':C_COLS+L+E+Q_COLS+SP}
CONTINUOUS=set(['B2_trend']+S+['RAI4','RAI5']+L+E)
PREPROCESS_LOG=[];FIT_DIAGNOSTICS=[];ATTEMPT_LOG=[];FIT_LOG=[]
PAIRS=[('MQ','MQS','Add core S beyond Q'),('MS','MQS','Add Q beyond core S'),('MQ','MS','Core S versus Q'),
 ('MQ','MQSplus','Add extended S beyond Q'),('MSplus','MQSplus','Add Q beyond extended S'),('MQ','MSplus','Extended S versus Q'),
 ('MLEQ','MLEQSplus','Add extended S given L/E/Q'),('MLESplus','MLEQSplus','Add Q given L/E/S'),('MLEQ','MLESplus','Extended S versus Q given L/E')]

def registry(outcome):
    return [(family,model) for family in (['PO','MN_L2','PPO'] if outcome=='N1' else ['PO','MN_L2']) for model in SPECS]

def fold_task(task):
    threadpool_limits(1)
    kind,outcome,repeat,fold,inner_fold,train,test=task
    global PREPROCESS_LOG,FIT_DIAGNOSTICS
    PREPROCESS_LOG=[];FIT_DIAGNOSTICS=[]
    predictions=[];scores=[]
    for family,model in registry(outcome):
        tag=f'{kind}/{outcome}/{repeat}/{fold}/{inner_fold}/{family}/{model}'
        p,native,diag=fit_predict(train,test,outcome,family,model,tag)
        if kind=='outer':predictions.extend(prediction_records(test,outcome,family,model,repeat,fold,p,native))
        else:
            loss=losses(y_three(test,outcome),p)['log_loss'] if p is not None else np.full(len(test),np.nan)
            scores.append(dict(outcome=outcome,outer_fold=fold,inner_fold=inner_fold,family=family,model=model,
                               n=len(test),valid=bool(np.isfinite(loss).all()),loss_sum=float(loss.sum())))
    return predictions,scores,FIT_DIAGNOSTICS,PREPROCESS_LOG

def collect_tasks(tasks,workers):
    predictions=[];scores=[];health=[];scaling=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fold_task,t) for t in tasks]
        for i,future in enumerate(as_completed(futures),1):
            p,s,h,z=future.result();predictions.extend(p);scores.extend(s);health.extend(h);scaling.extend(z)
            if i%5==0:print('Prediction fold tasks:',i,'/',len(tasks),flush=True)
    return predictions,scores,health,scaling

def calibration(frame,args):
    rows=[];bins=[]
    for outcome in ['N1','N3']:
        y=y_three(frame,outcome);n=len(y)
        for name,cols in [('CQ',C_COLS+Q_COLS),('CS',C_COLS+['CAU','AOR'])]:
            probabilities=np.full((2,n,3),np.nan)
            for repeat in range(2):
                split=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED+repeat)
                for train,test in split.split(np.zeros(n),y):
                    scaler=StandardScaler();a=scaler.fit_transform(frame.iloc[train][cols]);b=scaler.transform(frame.iloc[test][cols])
                    model=LogisticRegression(C=1.,solver='lbfgs',max_iter=1500,tol=1e-7)
                    model.fit(a,y[train])
                    if model.n_iter_.max()>=1500:raise RuntimeError('Calibration-model optimizer did not converge.')
                    probabilities[repeat,test]=model.predict_proba(b)
            if not np.isfinite(probabilities).all():raise RuntimeError('Incomplete calibration OOF predictions.')
            p=np.clip(probabilities.mean(axis=0)[:,2],1e-6,1-1e-6);event=(y==2).astype(int)
            logits=stats.logistic.ppf(p);fit=sm.GLM(event,sm.add_constant(logits),family=sm.families.Binomial()).fit()
            rows.append(dict(outcome=outcome,model=name,n=n,calibration_intercept=float(fit.params[0]),calibration_slope=float(fit.params[1])))
            groups=pd.qcut(pd.Series(p),q=5,duplicates='drop')
            for label,idx in pd.Series(np.arange(n)).groupby(groups,observed=True):
                ix=idx.to_numpy();bins.append(dict(outcome=outcome,model=name,bin=str(label),n=len(ix),mean_predicted=float(p[ix].mean()),observed_rate=float(event[ix].mean())))
    save_table(args.output,'05_prediction','calibration',pd.DataFrame(rows))
    save_table(args.output,'05_prediction','calibration_bins',pd.DataFrame(bins))

def run(args):
    frame,*_=load_analysis(args.data);phase='05_prediction'
    tasks=[];splits={};audit=[]
    for oi,outcome in enumerate(['N3','N1']):
        for repeat in range(3):
            split=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED+100*oi+repeat)
            for fold,(train,test) in enumerate(split.split(frame,frame[outcome].astype(int)),1):
                splits[(outcome,repeat,fold)]=(train,test)
                a=frame.iloc[train].reset_index(drop=True);b=frame.iloc[test].reset_index(drop=True)
                tasks.append(('outer',outcome,repeat,fold,0,a,b))
                audit.append(dict(outcome=outcome,repeat=repeat,fold=fold,n_train=len(a),n_test=len(b),overlap=len(set(train)&set(test))))
    predictions,_,health,scaling=collect_tasks(tasks,args.workers)
    oof=pd.DataFrame(predictions).sort_values(['outcome','family','model','repeat','respondent_id'])
    summary=[];loss_rows=[]
    for (outcome,family,model),part in oof.groupby(['outcome','family','model'],sort=True):
        eligible=len(part)==3*len(frame) and part.success.all()
        row=dict(outcome=outcome,family=family,model=model,n=len(frame),coverage=part.success.mean(),eligible_for_comparison=bool(eligible))
        if eligible:
            v=losses(part.y3,part[['p_0','p_1','p_2']].to_numpy());row.update({k+'_mean':float(a.mean()) for k,a in v.items()})
            base=part[['outcome','family','model','repeat','respondent_id']].copy()
            for k,values in v.items():base[k]=values
            loss_rows.extend(base.to_dict('records'))
        summary.append(row)
    summary=pd.DataFrame(summary);loss_table=pd.DataFrame(loss_rows)
    gains=[];rng=np.random.default_rng(SEED+6000)
    for outcome in ['N3','N1']:
        for family in sorted({f for f,m in registry(outcome)}):
            eligible=set(summary.loc[summary.outcome.eq(outcome)&summary.family.eq(family)&summary.eligible_for_comparison,'model'])
            for baseline,comparison,label in PAIRS:
                for metric in ['log_loss','Brier','RPS']:
                    row=dict(outcome=outcome,family=family,baseline=baseline,comparison=comparison,contrast=label,metric=metric,
                             gain=np.nan,CI95_low=np.nan,CI95_high=np.nan,status='Unavailable: incomplete coverage')
                    if baseline in eligible and comparison in eligible:
                        part=loss_table.loc[loss_table.outcome.eq(outcome)&loss_table.family.eq(family)]
                        a=part.loc[part.model.eq(baseline)].groupby('respondent_id')[metric].mean()
                        b=part.loc[part.model.eq(comparison)].groupby('respondent_id')[metric].mean()
                        d=(a-b).dropna().to_numpy();assert len(d)==len(frame)
                        boot=d[rng.integers(0,len(d),(args.bootstrap,len(d)))].mean(axis=1)
                        lo,hi=np.quantile(boot,[.025,.975]);row.update(gain=float(d.mean()),CI95_low=lo,CI95_high=hi,status='Conditional bootstrap of fixed OOF losses; no refitting')
                    gains.append(row)
    gains=pd.DataFrame(gains)
    save_table(args.output,phase,'candidate_metrics',summary)
    save_table(args.output,phase,'paired_predictive_gains',gains)
    save_table(args.output,phase,'primary_logloss_comparisons',gains.loc[gains.metric.eq('log_loss')&gains.contrast.isin(['Core S versus Q','Add core S beyond Q'])])
    save_table(args.output,phase,'fold_audit',pd.DataFrame(audit))
    # Nested selection uses the first outer repetition, never the outer test outcome.
    tasks=[];nested_audit=[]
    for oi,outcome in enumerate(['N3','N1']):
        for fold in range(1,6):
            tr,te=splits[(outcome,0,fold)];outer=frame.iloc[tr].reset_index(drop=True)
            splitter=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED+7000+100*oi+fold)
            for inner,(train,test) in enumerate(splitter.split(outer,outer[outcome].astype(int)),1):
                a=outer.iloc[train].reset_index(drop=True);b=outer.iloc[test].reset_index(drop=True)
                tasks.append(('inner',outcome,0,fold,inner,a,b))
                nested_audit.append(dict(outcome=outcome,outer_fold=fold,inner_fold=inner,
                    inner_overlap=len(set(train)&set(test)),outer_test_seen=len((set(a.respondent_id)|set(b.respondent_id))&set(frame.iloc[te].respondent_id))))
    _,inner,inner_health,inner_scaling=collect_tasks(tasks,args.workers)
    scores=pd.DataFrame(inner);choices=[];nested=[]
    for outcome in ['N3','N1']:
        for fold in range(1,6):
            part=scores.loc[scores.outcome.eq(outcome)&scores.outer_fold.eq(fold)]
            eligible=[]
            for (family,model),z in part.groupby(['family','model']):
                if len(z)==3 and z.valid.all():eligible.append(dict(family=family,model=model,inner_log_loss=float(z.loss_sum.sum()/z.n.sum()),slopes=len(SPECS[model])))
            if not eligible:raise RuntimeError('No complete inner-CV candidate.')
            best=min(eligible,key=lambda r:(r['inner_log_loss'],r['slopes'],r['family'],r['model']))
            test=oof.loc[oof.outcome.eq(outcome)&oof.repeat.eq(0)&oof.fold.eq(fold)&oof.family.eq(best['family'])&oof.model.eq(best['model'])]
            choices.append(dict(outcome=outcome,outer_fold=fold,**best,outer_prediction_complete=bool(test.success.all())))
            nested.append(test)
    nested=pd.concat(nested,ignore_index=True);nested_metrics=[]
    for outcome,t in nested.groupby('outcome'):
        complete=len(t)==len(frame) and t.success.all()
        row=dict(outcome=outcome,n=len(t),complete=bool(complete))
        if complete:row.update({key:float(value.mean()) for key,value in losses(t.y3,t[['p_0','p_1','p_2']].to_numpy()).items()})
        nested_metrics.append(row)
    save_table(args.output,phase,'nested_selection_metrics',pd.DataFrame(nested_metrics))
    save_table(args.output,phase,'nested_selected_candidates',pd.DataFrame(choices))
    save_table(args.output,phase,'nested_leakage_audit',pd.DataFrame(nested_audit))
    diagnostics=pd.DataFrame(health+inner_health)
    save_table(args.output,phase,'fit_health_summary',diagnostics.groupby(['outcome','family','model','success']).size().rename('fits').reset_index())
    cache=Path(args.repo)/'.work';cache.mkdir(exist_ok=True)
    for name,value in [('oof_predictions',oof),('oof_losses',loss_table),('cv_diagnostics',diagnostics),('cv_scaling',pd.DataFrame(scaling+inner_scaling)),('nested_inner_scores',scores)]:value.to_pickle(cache/(name+'.pkl'))
    calibration(frame,args)
    finish_phase(args.output,phase)
    return gains

def validate_X(X):
    X = X.astype(float)
    a = X.to_numpy()
    if not np.isfinite(a).all():
        raise ValueError('Design contains missing/nonfinite values.')
    if np.linalg.matrix_rank(a) != a.shape[1]:
        raise ValueError('Design matrix is not full rank.')
    if np.linalg.matrix_rank(np.column_stack([np.ones(len(X)), a])) != a.shape[1] + 1:
        raise ValueError('Explicit or implicit constant found in ordinal design.')
    return X

def result_health(result, p):
    cov = np.asarray(result.cov_params())
    eigen = np.linalg.eigvalsh(cov) if np.isfinite(cov).all() else np.array([np.nan])
    return bool(result.mle_retvals.get('converged', False) and np.isfinite(result.llf) and np.isfinite(result.bse).all() and np.all(np.asarray(result.bse) > 0) and np.all(eigen > 0) and (np.max(np.abs(np.asarray(result.params)[:p])) < 20) and (np.max(np.asarray(result.bse)[:p]) < 10))

def fit_ordered(data, outcome, columns, tag, audit=True, start=None):
    X = validate_X(data[columns])
    y = data[outcome].astype(int)
    if sorted(y.unique()) != list(range(1, int(y.max()) + 1)):
        raise ValueError('Outcome must retain consecutive observed ordered classes.')
    model = OrderedModel(y, X, distr='logit')
    attempts = []
    for method in ['bfgs', 'lbfgs']:
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                result = model.fit(method=method, maxiter=FIT_MAXITER, disp=False, start_params=start)
                healthy = result_health(result, len(columns))
                message = ' | '.join((str(w.message) for w in caught))
                attempts.append((result, healthy, method, message))
                start = np.asarray(result.params)
                record = {'tag': tag, 'optimizer': method, 'healthy': healthy, 'converged': bool(result.mle_retvals.get('converged', False)), 'llf': float(result.llf), 'seconds': time.perf_counter() - started, 'warnings': message, 'iterations_or_calls': int(result.mle_retvals.get('iterations', result.mle_retvals.get('fcalls', -1)))}
            except Exception as exc:
                record = {'tag': tag, 'optimizer': method, 'healthy': False, 'converged': False, 'llf': np.nan, 'seconds': time.perf_counter() - started, 'warnings': repr(exc), 'iterations_or_calls': -1}
        if audit:
            ATTEMPT_LOG.append(record)
        if attempts and attempts[-1][1]:
            break
    if not attempts:
        raise RuntimeError(f'All fits failed: {tag}; inspect fit_attempts.')
    good = [a for a in attempts if a[1]]
    result, healthy, method, message = max(good or attempts, key=lambda a: a[0].llf)
    result._phase5_fit_messages = ' | '.join((a[3] for a in attempts if a[3]))
    if audit:
        probs = np.asarray(result.model.predict(result.params, exog=X))
        grad = np.asarray(result.model.score(result.params))
        FIT_LOG.append({'tag': tag, 'outcome': outcome, 'n': len(data), 'slopes': len(columns), 'thresholds': len(result.params) - len(columns), 'parameters': len(result.params), 'optimizer': method, 'converged': bool(result.mle_retvals.get('converged', False)), 'numerically_healthy': healthy, 'llf': result.llf, 'AIC_within_same_outcome_only': result.aic, 'BIC_within_same_outcome_only': result.bic, 'max_abs_score_per_person': float(np.max(np.abs(grad)) / len(data)), 'max_abs_beta': float(np.max(np.abs(np.asarray(result.params)[:len(columns)]))), 'max_slope_SE': float(np.max(np.asarray(result.bse)[:len(columns)])), 'probability_sum_error': float(np.max(np.abs(probs.sum(axis=1) - 1))), 'min_probability': float(probs.min()), 'warnings': message})
    return (result, healthy)

def fit_ppo(data, outcome, cols, base):
    T = len(OUTCOME_LEVELS[outcome]) - 1
    varying = [c for c in cols if c.startswith(('CAU_z_', 'AOR_z_'))]
    common = [c for c in cols if c not in varying]
    X = data[common].to_numpy(float)
    Z = data[varying].to_numpy(float)
    n = len(data)
    q = len(common)
    r = len(varying)
    d = T + q + T * r
    y = data[outcome].to_numpy(int) - 1
    A = np.zeros((n, T, d))
    for j in range(T):
        A[:, j, j] = 1
        A[:, j, T:T + q] = -X
        A[:, j, T + q + j * r:T + q + (j + 1) * r] = -Z
    constraints = np.diff(A, axis=1).reshape(-1, d)

    def probabilities(theta):
        F = expit(A @ theta)
        return np.diff(np.column_stack([np.zeros(n), F, np.ones(n)]), axis=1)

    def objective(theta):
        p = probabilities(theta)[np.arange(n), y]
        return -np.log(np.maximum(p, 1e-300)).sum()

    def gradient(theta):
        F = expit(A @ theta)
        derivatives = F[:, :, None] * (1 - F[:, :, None]) * A
        extended = np.concatenate([np.zeros((n, 1, d)), derivatives, np.zeros((n, 1, d))], axis=1)
        dp = np.diff(extended, axis=1)[np.arange(n), y]
        p = probabilities(theta)[np.arange(n), y]
        return -(dp / np.maximum(p[:, None], 1e-300)).sum(axis=0)
    start = np.r_[base.model.transform_threshold_params(base.params)[1:-1], [base.params[c] for c in common], np.tile([base.params[c] for c in varying], T)]
    null = np.r_[stats.logistic.ppf([np.mean(y <= j) for j in range(T)]), np.zeros(q + T * r)]
    candidates = []
    attempts = []
    for label, initial in [('PO_start', start), ('null_start', null)]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fit = minimize(objective, initial, jac=gradient, method='SLSQP', constraints=[LinearConstraint(constraints, 1e-06, np.inf)], options={'maxiter': 1500, 'ftol': 1e-09})
        gap = float((constraints @ fit.x).min())
        prob = probabilities(fit.x)
        valid = bool(fit.success and gap >= 9.99e-07 and np.isfinite(fit.fun) and (prob.min() >= 0))
        attempts.append({'start': label, 'success': bool(fit.success), 'valid': valid, 'llf': -fit.fun, 'message': str(fit.message), 'warnings': ' | '.join((str(w.message) for w in caught))})
        if valid:
            candidates.append(fit)
    if not candidates:
        raise RuntimeError(str(attempts))
    fit = min(candidates, key=lambda x: x.fun)
    gap = float((constraints @ fit.x).min())
    info = approx_hess(fit.x, objective)
    eig = np.linalg.eigvalsh(info)
    interior = gap > 0.0001
    posdef = bool(np.all(eig > 1e-07))
    agreement = len(candidates) == 2 and abs(candidates[0].fun - candidates[1].fun) < 0.0001
    grad = float(np.max(np.abs(gradient(fit.x))) / n)
    infer = bool(interior and posdef and agreement and (grad < 0.0001))
    covariance = np.linalg.inv(info) if posdef else np.full((d, d), np.nan)
    return dict(theta=fit.x, cov=covariance, prob=probabilities(fit.x), llf=-fit.fun, k=d, common=common, varying=varying, T=T, q=q, r=r, inference_valid=infer, interior=interior, min_logit_gap=gap, min_information_eigenvalue=float(eig.min()), two_start_agreement=agreement, max_score_per_person=grad, attempts=attempts, gradient_error=float(np.max(np.abs(gradient(fit.x) - approx_fprime(fit.x, objective, centered=True)))))

def y_three(data, outcome):
    return data[outcome + '_3'].map({'Decreased': 0, 'Unchanged': 1, 'Increased': 2}).to_numpy(int)

def prepare_design(train, test, cols, outcome, tag):
    a = train.copy()
    b = test.copy()
    kept = []
    dropped = []
    zrows = []
    rank_matrix = np.ones((len(train), 1))
    for c in cols:
        x = train[c].to_numpy(float)
        mu = float(x.mean())
        sd = float(x.std(ddof=1))
        name = c
        if c in CONTINUOUS:
            name = f'{c}_z_{outcome}'
            if sd > 1e-12:
                a[name] = (train[c] - mu) / sd
                b[name] = (test[c] - mu) / sd
            else:
                a[name] = 0.0
                b[name] = 0.0
            zrows.append(dict(tag=tag, variable=c, mean=mu, sd=sd, n_train=len(train)))
        proposed = np.column_stack([rank_matrix, a[name].to_numpy(float)])
        if np.linalg.matrix_rank(proposed) > rank_matrix.shape[1]:
            kept.append(name)
            rank_matrix = proposed
        else:
            dropped.append(c)
    PREPROCESS_LOG.extend(zrows)
    return (a, b, kept, dropped)

def valid_probs(p, k):
    p = np.asarray(p, float)
    return bool(p.ndim == 2 and p.shape[1] == k and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all() and np.allclose(p.sum(1), 1, atol=1e-08))

def collapse_probs(p, outcome):
    return np.column_stack([p[:, :2].sum(1), p[:, 2], p[:, 3:].sum(1)]) if outcome == 'N3' else np.column_stack([p[:, 0], p[:, 1], p[:, 2:].sum(1)])

def predict_ppo(fit, test):
    T, q, r = (fit['T'], fit['q'], fit['r'])
    theta = fit['theta']
    X = test[fit['common']].to_numpy(float)
    Z = test[fit['varying']].to_numpy(float)
    eta = theta[:T][None, :] - (X @ theta[T:T + q])[:, None]
    if r:
        eta = eta - Z @ theta[T + q:].reshape(T, r).T
    if np.any(np.diff(eta, axis=1) < 0):
        raise ValueError('PPO threshold crossing on held-out covariates; no clipping or fallback')
    F = expit(eta)
    p = np.diff(np.column_stack([np.zeros(len(test)), F, np.ones(len(test))]), axis=1)
    if not valid_probs(p, T + 1):
        raise ValueError('Invalid PPO held-out probabilities')
    return p

def fit_predict(train, test, outcome, family, model, tag):
    start = time.perf_counter()
    record = dict(tag=tag, outcome=outcome, family=family, model=model, n_train=len(train), n_test=len(test), success=False)
    p3 = None
    native = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            a, b, cols, dropped = prepare_design(train, test, SPECS[model], outcome, tag)
            record.update(dropped_columns='; '.join(dropped), actual_slopes=len(cols))
            X = validate_X(a[cols])
            Xt = b[cols].astype(float)
            if family == 'MN_L2':
                res = LogisticRegression(C=RIDGE_C, solver='lbfgs', max_iter=2000, tol=1e-08)
                res.fit(X, y_three(a, outcome))
                if not np.array_equal(res.classes_, [0, 1, 2]) or np.max(res.n_iter_) >= 2000:
                    raise ValueError('Multinomial class/convergence check failed')
                p3 = res.predict_proba(Xt)
                record['optimizer_iterations'] = int(np.max(res.n_iter_))
            else:
                base, ok = fit_ordered(a, outcome, cols, tag, audit=False)
                record['PO_fit_messages'] = getattr(base, '_phase5_fit_messages', '')
                if not ok:
                    raise ValueError('Ordinal numerical-health check failed')
                if family == 'PO':
                    native = np.asarray(base.predict(Xt))
                else:
                    fit = fit_ppo(a, outcome, cols, base)
                    record.update(PPO_interior=fit['interior'], PPO_two_start_agreement=fit['two_start_agreement'], PPO_min_gap=fit['min_logit_gap'])
                    if not fit['inference_valid']:
                        raise ValueError('PPO boundary/numerical check failed; candidate unavailable for this fold')
                    native = predict_ppo(fit, b)
                if not valid_probs(native, len(OUTCOME_LEVELS[outcome])):
                    raise ValueError('Invalid native probabilities')
                p3 = collapse_probs(native, outcome)
            if not valid_probs(p3, 3):
                raise ValueError('Invalid three-class probabilities')
            record['success'] = True
            record['reason'] = 'OK'
        except Exception as exc:
            record['reason'] = repr(exc)
            p3 = None
            native = None
    record['warnings'] = ' | '.join((str(w.message) for w in caught))
    record['seconds'] = time.perf_counter() - start
    FIT_DIAGNOSTICS.append(record)
    return (p3, native, record)

def losses(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    one = np.eye(p.shape[1])[y]
    return {'log_loss': -np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)), 'Brier': ((p - one) ** 2).sum(1), 'RPS': ((np.cumsum(p, axis=1)[:, :-1] - np.cumsum(one, axis=1)[:, :-1]) ** 2).mean(1)}

def prediction_records(test, outcome, family, model, repeat, fold, p3, native):
    rows = []
    y = y_three(test, outcome)
    for i, (_, row) in enumerate(test.iterrows()):
        rr = dict(respondent_id=row.respondent_id, outcome=outcome, family=family, model=model, repeat=repeat, fold=fold, y3=int(y[i]), y_native=int(row[outcome]), success=p3 is not None)
        rr.update({f'p_{k}': float(p3[i, k]) if p3 is not None else np.nan for k in range(3)})
        rr.update({f'native_p_{k + 1}': float(native[i, k]) if native is not None and k < native.shape[1] else np.nan for k in range(5)})
        rows.append(rr)
    return rows
