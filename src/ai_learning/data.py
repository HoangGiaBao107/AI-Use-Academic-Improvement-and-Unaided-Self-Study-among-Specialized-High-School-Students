"""Exact codebook and complete-case analytical cohort."""
from pathlib import Path
import hashlib,io,re
import numpy as np
import pandas as pd
from scipy import stats

EXPECTED_HASH = "ded803a8b38c1e14b31835655c1ad2b54854e21ec05c191fdd5dd7bead471624"
C_COLS = ["grade_11","grade_12","male","A7_lower","A7_unknown"]
B1_LEVELS = ["3\u20135 times/week","At most 2 times/week","6\u201310 times/week","More than 10 times/week","Almost daily"]
B2_LEVELS = ["Under 10 minutes","10 to <20 minutes","20 to <40 minutes","40 to <60 minutes","60+ minutes"]
EXPANDED = ["AIL","TAG","AIC","RAI_peer3","RAI4","RAI5"]
BLOCKS = {b:[f"{b}{i}" for i in range(1,7 if b=="AIL" else 6)] for b in ["AIL","CAU","AOR","TAG","AIC","RAI","PLG"]}

def raw_table(path):
    raw = pd.read_csv(path, dtype='string', keep_default_na=False)
    headers = {}
    for column in raw:
        match = re.search(r'(?:^|\[)([A-Z]+\d+[a-d]?)\.', column)
        if not match:
            raise ValueError('Unrecognized CSV header: '+column)
        headers[match.group(1)] = column
    raw.columns = list(headers)
    raw = raw.apply(lambda c: c.str.replace(r'\s+', ' ', regex=True).str.strip())
    return raw, headers

def load_analysis(path):
    frame, digest = load_and_encode(path)
    raw, headers = raw_table(path)
    labels = {'Decreased significantly':1,'Decreased slightly':2,
              'Almost unchanged':3,'Increased slightly':4,'Increased significantly':5}
    for name in ['N1','N3']:
        original = raw.loc[frame.index,name].map(labels).astype(float)
        frame[name+'_original5'] = original
        frame[name] = original.map({1:1,2:1,3:2,4:3,5:4}) if name=='N1' else original
        frame[name+'_decline'] = original.le(2).astype(float)
        frame[name+'_3'] = pd.Categorical(original.map({1:'Decreased',2:'Decreased',3:'Unchanged',4:'Increased',5:'Increased'}),
                                           ['Decreased','Unchanged','Increased'],ordered=True)
    frame['N1_not_decline'] = frame.N1_decline.eq(0)
    frame['focal'] = (frame.N1_not_decline & frame.N3_decline.eq(1)).astype(float)
    frame['state'] = (2*frame.N1_decline+frame.N3_decline).astype(int)
    frame['Both_up'] = frame.N1_increase*frame.N3_increase
    frame['N1_up'] = frame.N1_increase
    frame['N3_up'] = frame.N3_increase
    frame['Academic_non_decline_selfstudy_decline'] = frame.focal
    for name in ['A1','A2','A3','A5','A7']:
        frame[name+'_label'] = raw.loc[frame.index,name]
    retained = raw.loc[frame.index].copy()
    frame = frame.reset_index(drop=True)
    retained = retained.reset_index(drop=True)
    assert len(frame)==323 and frame.Both_up.sum()==101
    return frame, retained, raw, headers, digest

def reference_scales(frame):
    return {name:(float(frame[name].mean()),float(frame[name].std(ddof=1)))
            for name in ['CAU','AOR','AOR_no2','AOR_no3']+EXPANDED}

def rcs(z, knots, derivative=False):
    a, b, c = knots
    power = 2 if derivative else 3
    h = np.maximum(z - a, 0) ** power - np.maximum(z - b, 0) ** power * (c - a) / (c - b) + np.maximum(z - c, 0) ** power * (b - a) / (c - b)
    return h * (3 if derivative else 1) / (c - a) ** 2

def bh_adjust(pvalues):
    p = np.asarray(pvalues, float)
    finite = np.isfinite(p)
    a = np.where(finite, p, 1.0)
    if not ((a >= 0) & (a <= 1)).all():
        raise ValueError('p-value outside [0, 1].')
    order = np.argsort(a, kind='stable')
    q = np.minimum.accumulate((a[order] * len(a) / np.arange(1, len(a) + 1))[::-1])[::-1]
    result = np.empty(len(a))
    result[order] = np.minimum(q, 1)
    result[~finite] = np.nan
    assert np.allclose(np.nan_to_num(result, nan=1.0), stats.false_discovery_control(a, method='bh'))
    return result

def load_and_encode(path):
    source = Path(path).read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    if source_hash != EXPECTED_HASH:
        raise ValueError('Input hash does not match the reviewed English CSV.')
    original = pd.read_csv(io.BytesIO(source), dtype='string', keep_default_na=False)

    def code(column):
        match = re.search('\\[((?:AIL|CAU|AOR|TAG|AIC|RAI|PLG)\\d+)\\.', column)
        match = match or re.match('^([ABN]\\d+[a-d]?)\\.', column.strip())
        if not match:
            raise ValueError('Unrecognized variable header: ' + column)
        return match.group(1)
    text = original.copy()
    text.columns = [code(c) for c in original]
    assert text.columns.is_unique
    for c in text:
        text[c] = text[c].str.replace('\xa0', ' ', regex=False).str.replace('\\s+', ' ', regex=True).str.strip().replace('', pd.NA)
    ids = pd.Series([f'R{i:04d}' for i in range(1, len(text) + 1)], index=text.index)
    blocks = {s: [f'{s}{i}' for i in range(1, 7 if s == 'AIL' else 6)] for s in ['AIL', 'CAU', 'AOR', 'TAG', 'AIC', 'RAI', 'PLG']}
    likert = [v for group in blocks.values() for v in group]
    freq = {f'{i} \u2013 {v}': i for i, v in enumerate(['Never', 'Rarely', 'Sometimes', 'Often', 'Always'], 1)}
    agreement = {f'{i} \u2013 {v}': i for i, v in enumerate(['Strongly disagree', 'Disagree', 'Neutral / Not sure', 'Agree', 'Strongly agree'], 1)}
    frame = pd.DataFrame({'respondent_id': ids})
    for c in likert:
        mapping = freq if c.startswith(('CAU', 'AOR')) else agreement
        bad = text[c].notna() & ~text[c].isin(mapping)
        if bad.any():
            raise ValueError('Unrecognized Likert label: ' + c)
        frame[c] = text[c].map(mapping).astype(float)
    for s, columns in blocks.items():
        frame[s] = frame[columns].mean(axis=1).where(frame[columns].notna().all(axis=1))
    for name, columns in {'AOR_no2': ['AOR1', 'AOR3', 'AOR4', 'AOR5'], 'AOR_no3': ['AOR1', 'AOR2', 'AOR4', 'AOR5'], 'RAI_peer3': ['RAI1', 'RAI2', 'RAI3']}.items():
        frame[name] = frame[columns].mean(axis=1).where(frame[columns].notna().all(axis=1))
    outcomes = {'Decreased significantly', 'Decreased slightly', 'Almost unchanged', 'Increased slightly', 'Increased significantly'}
    for name in ['N1', 'N3']:
        assert text[name].dropna().isin(outcomes).all()
        frame[name + '_increase'] = text[name].isin(['Increased slightly', 'Increased significantly']).astype(float).where(text[name].notna())
    frame['grade_11'] = text.A1.eq('11').astype(float)
    frame['grade_12'] = text.A1.eq('12').astype(float)
    frame['male'] = text.A3.map({'0': 0, '1': 1}).astype(float)
    frame['A7_lower'] = text.A7.isin(['Fair', 'Pass', 'Not passed']).astype(float)
    frame['A7_unknown'] = text.A7.isin(["Don't remember", "Don't remember / Prefer not to answer"]).astype(float)
    b1 = text.B1.replace({'Less than once/week': 'At most 2 times/week', '1\u20132 times/week': 'At most 2 times/week'})
    assert b1.isin(B1_LEVELS).all()
    for k, level in enumerate(B1_LEVELS[1:], 1):
        frame[f'B1_dummy_{k}'] = b1.eq(level).astype(float)
    assert text.B2.isin(B2_LEVELS).all()
    frame['B2_trend'] = text.B2.map({v: i for i, v in enumerate(B2_LEVELS, 1)}).astype(float)
    for k in range(2, 6):
        frame[f'B2_dummy_{k}'] = frame.B2_trend.eq(k).astype(float)
    values = frame[likert]

    def longest(a):
        best = run = 0
        last = np.nan
        for value in a:
            run = run + 1 if np.isfinite(value) and value == last else 1 if np.isfinite(value) else 0
            best = max(best, run)
            last = value
        return best
    flat = pd.DataFrame({s: frame[c].notna().all(axis=1) & frame[c].std(axis=1, ddof=0).eq(0) for s, c in blocks.items()}).sum(axis=1)
    straight = values.notna().all(axis=1) & values.std(axis=1, ddof=0).eq(0) | values.apply(lambda x: longest(x.to_numpy(float)), axis=1).ge(15) | flat.ge(5)
    encoded = np.column_stack([pd.factorize(text[c], sort=True)[0] for c in text])
    near = np.zeros(len(frame), bool)
    for i in range(len(frame)):
        valid = (encoded >= 0) & (encoded[i] >= 0)
        denominator = valid.sum(axis=1)
        same = ((encoded == encoded[i]) & valid).sum(axis=1)
        similarity = np.divide(same, denominator, out=np.full(len(frame), np.nan), where=denominator >= np.ceil(0.9 * encoded.shape[1]))
        similarity[i] = np.nan
        near[i] = np.nanmax(similarity) >= 0.9
    keys = values.fillna(-1).astype(int).astype(str).agg('|'.join, axis=1)
    demo = text[['A1', 'A2', 'A3']].fillna('(Missing)').agg('|'.join, axis=1)
    purposes = text.B5.map(lambda v: len(set((t.strip() for t in re.split('[;,]', str(v)) if t.strip()))) if pd.notna(v) else np.nan)
    flags = pd.DataFrame({'straightline': straight, 'exact_raw': original.duplicated(keep=False), 'exact_normalized': text.duplicated(keep=False), 'near_duplicate': near, 'duplicate_likert': keys.duplicated(keep=False), 'duplicate_likert_demo': (keys + '#' + demo).duplicated(keep=False), 'B5_over_limit': purposes.gt(5)})
    flags['any_review'] = flags.any(axis=1)
    frame['straightline'] = flags.straightline
    frame['any_review'] = flags.any_review
    needed = C_COLS + [f'B1_dummy_{i}' for i in range(1, 5)] + ['B2_trend', 'CAU', 'AOR', 'N1_increase', 'N3_increase']
    frame = frame.loc[frame[needed].notna().all(axis=1)].copy()
    assert len(frame) == 323
    assert frame.N1_increase.sum() == 205 and frame.N3_increase.sum() == 122
    assert frame.straightline.sum() == 52 and frame.any_review.sum() == 122
    assert 'N2_increase' not in frame
    return (frame, source_hash)
