"""Leakage-safe preprocessing for network flow datasets."""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, RobustScaler, StandardScaler


def _read_all_csv(raw_dir, max_rows=None):
    files = list(Path(raw_dir).rglob('*.csv'))
    if not files:
        raise FileNotFoundError(f'No CSV files found under {raw_dir}')
    dfs = []
    remaining = max_rows
    for f in files:
        n = None if remaining is None else max(0, remaining)
        if n == 0:
            break
        for enc in ('utf-8', 'latin-1', 'cp1252'):
            try:
                df = pd.read_csv(f, low_memory=False, nrows=n, encoding=enc)
                break
            except (UnicodeDecodeError, ValueError):
                continue
        df.columns = [str(c).strip() for c in df.columns]
        dfs.append(df)
        if remaining is not None:
            remaining -= len(df)
    return pd.concat(dfs, ignore_index=True)


def _clean_inf_nan(df):
    return df.replace([np.inf, -np.inf], np.nan)


def _row_hash_frame(X):
    return pd.util.hash_pandas_object(X.astype(str), index=False).astype('uint64').astype(str)


def prepare_dataset(cfg, force=False):
    name = cfg['dataset']['name']
    out = Path(cfg['dataset']['processed_dir'])
    out.mkdir(parents=True, exist_ok=True)
    done = out / 'done.json'
    if done.exists() and not force:
        return out

    df = _read_all_csv(cfg['dataset']['raw_dir'], cfg['dataset'].get('max_rows'))
    df = _clean_inf_nan(df)
    label_col = cfg['dataset']['label_column']
    if label_col not in df.columns:
        raise ValueError(
            f'label_column {label_col} not found. Available columns: {list(df.columns)[:50]}...'
        )
    drop = [c for c in cfg['dataset'].get('drop_columns', []) if c in df.columns]
    df = df.drop(columns=drop)
    before = len(df)
    if cfg['preprocess'].get('remove_duplicates_before_split', True):
        df = df.drop_duplicates().reset_index(drop=True)
    dup_removed = before - len(df)

    y_raw = df[label_col].astype(str).fillna('missing')
    X = df.drop(columns=[label_col])
    nunique = X.nunique(dropna=True)
    keep = nunique[nunique > 1].index.tolist()
    X = X[keep]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        train_size=cfg['split']['train_size'],
        random_state=cfg['split']['random_state'],
        stratify=y,
    )
    val_rel = cfg['split']['val_size'] / (cfg['split']['val_size'] + cfg['split']['test_size'])
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        train_size=val_rel,
        random_state=cfg['split']['random_state'],
        stratify=y_tmp,
    )

    nunique_train = X_train.nunique(dropna=True)
    keep = nunique_train[nunique_train > 1].index.tolist()
    X_train = X_train[keep]
    X_val = X_val[keep]
    X_test = X_test[keep]

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    scaler = RobustScaler() if cfg['preprocess'].get('scaler') == 'robust' else StandardScaler()
    num_steps = [('imputer', SimpleImputer(strategy=cfg['preprocess'].get('missing_strategy', 'median')))]
    num_steps.append(('scaler', scaler))
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy=cfg['preprocess'].get('categorical_strategy', 'most_frequent'))),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])
    pre = ColumnTransformer(
        [('num', Pipeline(num_steps), num_cols), ('cat', cat_pipe, cat_cols)],
        remainder='drop',
    )

    clip_info = None
    if cfg['preprocess'].get('quantile_clip', True) and num_cols:
        qs = cfg['preprocess'].get('clip_quantiles', [0.001, 0.999])
        lo = X_train[num_cols].quantile(qs[0])
        hi = X_train[num_cols].quantile(qs[1])
        clip_info = {'lo': lo.to_dict(), 'hi': hi.to_dict()}
        for Xp in [X_train, X_val, X_test]:
            Xp[num_cols] = Xp[num_cols].astype('float64').clip(lower=lo, upper=hi, axis=1)

    Xtr = pre.fit_transform(X_train)
    Xva = pre.transform(X_val)
    Xte = pre.transform(X_test)
    feature_names = num_cols + cat_cols
    selector = None
    fs = cfg['preprocess'].get('feature_selection', {})
    if fs.get('enabled', False):
        k = min(int(fs.get('k_best', 64)), Xtr.shape[1])
        selector = SelectKBest(mutual_info_classif, k=k)
        Xtr = selector.fit_transform(Xtr, y_train)
        Xva = selector.transform(Xva)
        Xte = selector.transform(Xte)

    Xtr = np.nan_to_num(Xtr.astype('float32'), nan=0.0, posinf=0.0, neginf=0.0)
    Xva = np.nan_to_num(Xva.astype('float32'), nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(Xte.astype('float32'), nan=0.0, posinf=0.0, neginf=0.0)

    smote_cfg = cfg['preprocess'].get('smote', {})
    if smote_cfg.get('enabled', False):
        from imblearn.over_sampling import RandomOverSampler, SMOTE
        from imblearn.under_sampling import RandomUnderSampler

        unique_c, counts_c = np.unique(y_train, return_counts=True)
        k_n = int(smote_cfg.get('k_neighbors', 5))
        max_per_class = int(smote_cfg.get('max_per_class', 10000))
        undersample_majority = smote_cfg.get('undersample_majority', False)
        print(f'  [SMOTE] before: {len(y_train)} samples, min={counts_c.min()}, max={counts_c.max()}')

        if undersample_majority:
            target = min(int(counts_c.max()), max_per_class)
            over = {int(c): target for c, cnt in zip(unique_c, counts_c) if int(cnt) > target}
            if over:
                rus = RandomUnderSampler(sampling_strategy=over, random_state=42)
                Xtr, y_train = rus.fit_resample(Xtr, y_train)
                unique_c, counts_c = np.unique(y_train, return_counts=True)

        target = max_per_class
        tiny = {int(c): k_n + 2 for c, cnt in zip(unique_c, counts_c) if int(cnt) <= k_n}
        if tiny:
            ros = RandomOverSampler(sampling_strategy=tiny, random_state=42)
            Xtr, y_train = ros.fit_resample(Xtr, y_train)
            unique_c, counts_c = np.unique(y_train, return_counts=True)

        min_real = int(smote_cfg.get('min_samples_for_smote', 50))
        smote_strategy = {
            int(c): target for c, cnt in zip(unique_c, counts_c)
            if int(cnt) < target and int(cnt) >= min_real
        }
        if smote_strategy:
            sm = SMOTE(sampling_strategy=smote_strategy, k_neighbors=k_n, random_state=42)
            Xtr, y_train = sm.fit_resample(Xtr, y_train)
        skipped = [int(c) for c, cnt in zip(unique_c, counts_c) if int(cnt) < min_real]
        if skipped:
            print(f'  [SMOTE] skipped classes with <{min_real} samples: {skipped}')
        unique_c, counts_c = np.unique(y_train, return_counts=True)
        print(
            f'  [SMOTE] after:  {len(y_train)} samples, {len(unique_c)} classes, '
            f'all={counts_c.min()}-{counts_c.max()}'
        )

    np.save(out / 'X_train.npy', Xtr)
    np.save(out / 'X_val.npy', Xva)
    np.save(out / 'X_test.npy', Xte)
    np.save(out / 'y_train.npy', y_train)
    np.save(out / 'y_val.npy', y_val)
    np.save(out / 'y_test.npy', y_test)
    joblib.dump(
        {
            'preprocessor': pre,
            'label_encoder': le,
            'selector': selector,
            'clip_info': clip_info,
            'num_cols': num_cols,
            'cat_cols': cat_cols,
            'feature_names': feature_names,
        },
        out / 'preprocess.joblib',
    )

    htr = set(_row_hash_frame(X_train))
    hva = set(_row_hash_frame(X_val))
    hte = set(_row_hash_frame(X_test))
    tv_overlap = len(htr & hva)
    tt_overlap = len(htr & hte)
    vt_overlap = len(hva & hte)
    for split_name, count in [
        ('train_val', tv_overlap),
        ('train_test', tt_overlap),
        ('val_test', vt_overlap),
    ]:
        if count > 0:
            print(
                f'NOTICE: {count} identical feature-hash rows in {split_name} '
                f'(possible label noise: same features, different labels).'
            )

    report = {
        'dataset': name,
        'rows_after_clean': int(len(df)),
        'duplicates_removed': int(dup_removed),
        'n_features_transformed': int(Xtr.shape[1]),
        'classes': le.classes_.tolist(),
        'train_val_overlap': tv_overlap,
        'train_test_overlap': tt_overlap,
        'val_test_overlap': vt_overlap,
    }
    (out / 'leakage_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    done.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return out


def load_arrays(processed_dir, split):
    p = Path(processed_dir)
    return np.load(p / f'X_{split}.npy'), np.load(p / f'y_{split}.npy')
