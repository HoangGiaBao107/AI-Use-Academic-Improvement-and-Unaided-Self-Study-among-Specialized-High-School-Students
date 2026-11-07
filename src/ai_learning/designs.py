"""Fixed association specifications retained from the report."""
import numpy as np
from .data import rcs,EXPANDED
SCENARIOS = ["primary","no_A7","A7_Good_only","AOR_no2","AOR_no3","no_straightline","no_review_flags","B2_categorical","expanded_constructs","nonlinear_scores"]
CAU_SCENARIOS = SCENARIOS[:-1]+["nonlinear_CAU"]
OUTCOMES = ["Both_up","N1_up","N3_up","Academic_non_decline_selfstudy_decline"]

def make_aor_design(frame, scenario, scales):
    d = frame.copy()
    if scenario == 'no_straightline':
        d = d.loc[~d.straightline].copy()
    if scenario == 'no_review_flags':
        d = d.loc[~d.any_review].copy()
    if scenario == 'A7_Good_only':
        d = d.loc[d.A7_label.eq('Good')].copy()
    aor = scenario if scenario in ['AOR_no2', 'AOR_no3'] else 'AOR'
    names = ['Intercept', 'grade_11', 'grade_12', 'male']
    if scenario not in ['no_A7', 'A7_Good_only']:
        names += ['A7_lower', 'A7_unknown']
    names += [f'B1_dummy_{i}' for i in range(1, 5)]
    names += [f'B2_dummy_{i}' for i in range(2, 6)] if scenario == 'B2_categorical' else ['B2_trend']
    d['Intercept'] = 1.0
    if scenario == 'expanded_constructs':
        for c in EXPANDED:
            d[c + '_z'] = (d[c] - scales[c][0]) / scales[c][1]
        names += [c + '_z' for c in EXPANDED]
    d['CAU_z'] = (d.CAU - scales['CAU'][0]) / scales['CAU'][1]
    d['AOR_z'] = (d[aor] - scales[aor][0]) / scales[aor][1]
    names += ['CAU_z', 'AOR_z']
    knots = {}
    if scenario == 'nonlinear_scores':
        for c in ['CAU', 'AOR']:
            z = (frame[c] - scales[c][0]) / scales[c][1]
            knots[c] = np.quantile(z, [0.1, 0.5, 0.9])
            d[c + '_rcs'] = rcs(d[c + '_z'].to_numpy(), knots[c])
            names.append(c + '_rcs')
    if d[names].isna().any().any():
        raise ValueError('Missing predictor in specification: ' + scenario)
    X = d[names].to_numpy(float)
    protected = [names.index('CAU_z'), names.index('AOR_z')]
    if 'AOR_rcs' in names:
        protected += [names.index('AOR_rcs'), names.index('CAU_rcs')]
    X2, X3 = (X.copy(), X.copy())
    for value, xx in [(2.0, X2), (3.0, X3)]:
        z = (value - scales[aor][0]) / scales[aor][1]
        xx[:, names.index('AOR_z')] = z
        if 'AOR_rcs' in names:
            xx[:, names.index('AOR_rcs')] = rcs(np.repeat(z, len(xx)), knots['AOR'])
    tested = [names.index('AOR_z')]
    if 'AOR_rcs' in names:
        tested.append(names.index('AOR_rcs'))
    return dict(X=X, X2=X2, X3=X3, names=names, tested=tested, protected=protected, focal=d.focal.to_numpy(float), n3=d.N3_decline.to_numpy(float), eligible=d.N1_not_decline.to_numpy(bool), states=d.state.to_numpy(int), indices=d.index.to_numpy(int), aor_scale=scales[aor][1])

def make_cau_design(frame, scenario, scales):
    d = frame.copy()
    if scenario == 'no_straightline':
        d = d.loc[~d.straightline].copy()
    if scenario == 'no_review_flags':
        d = d.loc[~d.any_review].copy()
    if scenario == 'A7_Good_only':
        d = d.loc[d.A7_label.eq('Good')].copy()
    d['Intercept'] = 1.0
    names = ['Intercept', 'grade_11', 'grade_12', 'male']
    if scenario not in ['no_A7', 'A7_Good_only']:
        names += ['A7_lower', 'A7_unknown']
    names += [f'B1_dummy_{i}' for i in range(1, 5)]
    names += [f'B2_dummy_{i}' for i in range(2, 6)] if scenario == 'B2_categorical' else ['B2_trend']
    aor = scenario if scenario in ['AOR_no2', 'AOR_no3'] else 'AOR'
    for c, source in [('CAU', 'CAU'), ('AOR', aor)]:
        d[c + '_z'] = (d[source] - scales[source][0]) / scales[source][1]
        names.append(c + '_z')
    if scenario == 'expanded_constructs':
        for c in ['AIL', 'TAG', 'AIC', 'RAI_peer3', 'RAI4', 'RAI5']:
            d[c + '_z'] = (d[c] - scales[c][0]) / scales[c][1]
            names.append(c + '_z')
    knots = {}
    if scenario == 'nonlinear_CAU':
        for c in ['CAU', 'AOR']:
            z = (frame[c] - scales[c][0]) / scales[c][1]
            knots[c] = np.quantile(z, [0.1, 0.5, 0.9])
            d[c + '_rcs'] = rcs(d[c + '_z'].to_numpy(), knots[c])
            names.append(c + '_rcs')
    X = d[names].to_numpy(float)
    tested = [names.index('CAU_z')]
    protected = [names.index('CAU_z'), names.index('AOR_z')]
    if scenario == 'nonlinear_CAU':
        tested += [names.index('CAU_rcs')]
        protected += [names.index('CAU_rcs'), names.index('AOR_rcs')]
    X3, X4 = (X.copy(), X.copy())
    for value, xx in [(3, X3), (4, X4)]:
        z = (value - scales['CAU'][0]) / scales['CAU'][1]
        xx[:, names.index('CAU_z')] = z
        if 'CAU_rcs' in names:
            xx[:, names.index('CAU_rcs')] = rcs(np.repeat(z, len(xx)), knots['CAU'])
    return dict(X=X, X3=X3, X4=X4, tested=tested, protected=protected, Y=d[OUTCOMES].to_numpy(float))
