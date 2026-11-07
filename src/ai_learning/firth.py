"""Firth logistic fitting, fixed-design tests, and profile intervals."""
import numpy as np
from scipy import stats,linalg
from scipy.optimize import minimize,brentq,root
from scipy.special import expit

def components(beta, X, y, weights):
    eta = X @ beta
    mu = expit(eta)
    variance = np.exp(-np.logaddexp(0, eta) - np.logaddexp(0, -eta))
    v = weights * variance
    info = X.T @ (v[:, None] * X)
    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0 or not np.isfinite(logdet):
        raise FloatingPointError('Singular information')
    inverse = np.linalg.inv(info)
    h = v * np.einsum('ij,jk,ik->i', X, inverse, X)
    ll = float(weights @ (y * eta - np.logaddexp(0, eta)))
    return (ll + 0.5 * logdet, X.T @ (weights * (y - mu) + h * (0.5 - mu)), info, mu)

def fit_firth(X, y, weights=None, start=None, fixed=None, protected=()):
    n, p = X.shape
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    fixed = dict(fixed or {})
    _, R, pivot = linalg.qr(X[w > 0], mode='economic', pivoting=True)
    rank = int(np.sum(abs(np.diag(R)) > 1e-10 * max(1, abs(R[0, 0]))))
    cols = np.sort(pivot[:rank])
    dropped = sorted(set(range(p)) - set(cols))
    if set(dropped) & (set(protected) | set(fixed)):
        raise RuntimeError('A resample makes a focal predictor unidentifiable.')
    A = X[:, cols]
    positions = {old: new for new, old in enumerate(cols)}
    ff = {positions[j]: value for j, value in fixed.items()}
    free = np.array([j for j in range(rank) if j not in ff], int)
    initial = np.zeros(rank) if start is None else np.asarray(start)[cols].copy()
    for j, value in ff.items():
        initial[j] = value

    def unpack(v):
        b = initial.copy()
        b[free] = v
        return b

    def objective(v):
        return -components(unpack(v), A, y, w)[0]

    def gradient(v):
        return -components(unpack(v), A, y, w)[1][free]
    options = [{'method': 'BFGS', 'options': {'gtol': 1e-08, 'maxiter': 400}}, {'method': 'L-BFGS-B', 'options': {'ftol': 1e-14, 'gtol': 1e-09, 'maxiter': 1200, 'maxls': 50}}]
    healthy = False
    for attempt, opt in enumerate(options):
        result = minimize(objective, initial[free], jac=gradient, **opt)
        b = unpack(result.x)
        pll, score, info, mu = components(b, A, y, w)
        score_max = float(np.max(abs(score[free])))
        healthy = np.isfinite(pll) and score_max < 5e-06 and (np.linalg.eigvalsh(info).min() > 1e-12)
        if healthy:
            break
        initial = b
    if not healthy:
        previous_pll = pll
        refinement = root(gradient, b[free], method='hybr', options={'xtol': 1e-10})
        candidate = unpack(refinement.x)
        cpll, cscore, cinfo, cmu = components(candidate, A, y, w)
        cmax = float(np.max(abs(cscore[free])))
        healthy = np.isfinite(cpll) and cmax < 5e-06 and (np.linalg.eigvalsh(cinfo).min() > 1e-12) and (cpll >= previous_pll - 1e-08)
        if healthy:
            b, pll, score, info, mu, score_max = (candidate, cpll, cscore, cinfo, cmu, cmax)
            attempt = 2
    if not healthy:
        raise RuntimeError('Firth score check failed: ' + str(score_max))
    full = np.zeros(p)
    full[cols] = b
    return dict(beta=full, pll=float(pll), info=info, cols=cols, mu=mu, X=X, y=y, weights=w, protected=protected, score_max=score_max, dropped=dropped, fallback=bool(attempt))

def test_firth(fit, tested):
    null = fit_firth(fit['X'], fit['y'], fit['weights'], fit['beta'], {j: 0.0 for j in tested}, fit['protected'])
    lr = 2 * (fit['pll'] - null['pll'])
    if lr < -1e-06:
        raise RuntimeError('Constrained likelihood exceeds the full likelihood.')
    lr = max(0.0, float(lr))
    return (lr, float(stats.chi2.sf(lr, len(tested))), null)

def profile_ci(fit, j):
    center = float(fit['beta'][j])
    position = list(fit['cols']).index(j)
    se = float(np.sqrt(np.linalg.inv(fit['info'])[position, position]))
    cache = {}
    cutoff = stats.chi2.ppf(0.95, 1)

    def target(value):
        if value not in cache:
            f = fit_firth(fit['X'], fit['y'], fit['weights'], fit['beta'], {j: float(value)}, fit['protected'])
            cache[value] = 2 * (fit['pll'] - f['pll']) - cutoff
        return cache[value]
    bounds = []
    for direction in [-1, 1]:
        step = max(0.25, se)
        for _ in range(18):
            edge = center + direction * step
            if target(edge) > 0:
                break
            step *= 1.6
        else:
            raise RuntimeError('Could not bracket a profile endpoint.')
        bounds.append(float(brentq(target, *sorted([center, edge]), xtol=1e-07)))
    residual = max((abs(target(v)) for v in bounds))
    return (bounds, float(residual))
