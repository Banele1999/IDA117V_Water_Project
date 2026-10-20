
import os, json, random, warnings
from pathlib import Path

import numpy as np, pandas as pd, matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    learning_curve
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from scipy.stats import binomtest
import joblib


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
(ROOT / 'models').mkdir(exist_ok=True)

DATA = ROOT / 'data' / 'ghs-2023-hhold-v1.csv'

if not DATA.exists():
    raise FileNotFoundError(
        'Place the supplied GHS CSV at ' + str(DATA)
    )

df = pd.read_csv(DATA, low_memory=False)

TARGET = 'wat_inte_12mth'

# Restrict to observed yes/no responses; do not reinterpret not-applicable as no.
df = df[df[TARGET].isin(['1. Yes', '2. No'])].copy()
df['target'] = (df[TARGET] == '1. Yes').astype(int)

# No direct water-interruption follow-up variables or identifier/weight in predictors.
FEATURES = [
    'prov',
    'geotype',
    'hsg_maind',
    'hsg_tenure',
    'wat_drinkwat',
    'san_toil',
    'eng_access',
    'swr_rub',
    'head_age',
    'head_sex',
    'hholdsz',
    'fin_inc_main'
]

FEATURES = [x for x in FEATURES if x in df]

X = df[FEATURES].replace({
    '9. Unspecified': np.nan,
    '99. Unspecified': np.nan
})

y = df.target

# Survey categories are strings, even where they start with numerical codes.
for c in FEATURES:
    X[c] = X[c].astype('string').fillna('Missing').astype(str)


X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.25,
    random_state=SEED,
    stratify=y_trainval
)


def preprocessor(cols):
    return ColumnTransformer([
        (
            'categorical',
            OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            cols
        )
    ], remainder='drop')


def pipe(est, cols=FEATURES):
    return Pipeline([
        ('pre', preprocessor(cols)),
        ('model', est)
    ])


models = {
    'Dummy': pipe(
        DummyClassifier(strategy='most_frequent')
    ),
    'Logistic Regression': pipe(
        LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=SEED
        )
    ),
    'Random Forest': pipe(
        RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=5,
            class_weight='balanced_subsample',
            random_state=SEED,
            n_jobs=-1
        )
    )
}


# Two distinct tabular neural architectures: dense MLP and residual dense network.
import torch
from torch import nn

torch.manual_seed(SEED)
torch.set_num_threads(2)

from neural import NeuralNet


def train_neural(residual=False, dropout=0.25, epochs=35):
    pp = preprocessor(FEATURES)

    a = pp.fit_transform(X_train).astype('float32')
    b = pp.transform(X_val).astype('float32')

    model = NeuralNet(a.shape[1], residual, dropout)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.001
    )

    loss_fn = nn.BCEWithLogitsLoss()
    best = float('inf')
    patience = 0
    history = []
    beststate = None

    for epoch in range(epochs):
        model.train()

        order = np.random.default_rng(SEED + epoch).permutation(len(a))
        losses = []

        for ix in np.array_split(
            order,
            max(1, int(np.ceil(len(a) / 256)))
        ):
            xb = torch.from_numpy(a[ix])
            yb = torch.from_numpy(
                y_train.to_numpy(dtype='float32')[ix]
            )

            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

            losses.append(loss.item())

        model.eval()

        with torch.no_grad():
            vl = loss_fn(
                model(torch.from_numpy(b)),
                torch.tensor(y_val.to_numpy(dtype='float32'))
            ).item()

        history.append((epoch + 1, float(np.mean(losses)), vl))

        if vl < best - 1e-4:
            best = vl
            beststate = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1

        if patience >= 6:
            break

    model.load_state_dict(beststate)

    return pp, model, history


def predict_neural(pp, model, x):
    model.eval()

    with torch.no_grad():
        return torch.sigmoid(
            model(
                torch.from_numpy(
                    pp.transform(x).astype('float32')
                )
            )
        ).numpy()


def scores(name, truth, p):
    pred = (p >= 0.5).astype(int)

    return dict(
        model=name,
        accuracy=accuracy_score(truth, pred),
        macro_f1=f1_score(truth, pred, average='macro'),
        weighted_f1=f1_score(truth, pred, average='weighted'),
        precision=precision_score(truth, pred, zero_division=0),
        recall=recall_score(truth, pred),
        roc_auc=roc_auc_score(truth, p),
        pr_auc=average_precision_score(truth, p)
    )


# Dataset audit and five exploratory figures.
audit = pd.DataFrame({
    'column': df.columns,
    'missing_pct': df.isna().mean().values * 100,
    'unique': df.nunique(dropna=True).values
})

audit.to_csv(OUT / 'data_dictionary_audit.csv', index=False)


fig, ax = plt.subplots()
y.value_counts().sort_index().plot.bar(ax=ax)
ax.set_xticklabels(
    ['No interruption', 'Interruption'],
    rotation=0
)
ax.set_ylabel('Households')
ax.set_title('Reported water interruptions (eligible households)')
fig.tight_layout()
fig.savefig(OUT / 'eda_target.png')
plt.close(fig)


fig, ax = plt.subplots()
audit.sort_values(
    'missing_pct',
    ascending=False
).head(15).set_index('column').missing_pct.plot.bar(ax=ax)

ax.set_ylabel('Missing (%)')
ax.set_title('Top 15 columns by missingness')
fig.tight_layout()
fig.savefig(OUT / 'eda_missing.png')
plt.close(fig)


for col in ['prov', 'geotype', 'wat_drinkwat']:
    tab = df.groupby(col)['target'].agg(['mean', 'size']).sort_values('mean')

    fig, ax = plt.subplots(figsize=(9, 5))
    tab['mean'].mul(100).plot.barh(ax=ax)

    ax.set_xlabel(
        'Interruption prevalence among eligible households (%)'
    )
    ax.set_title('Water interruptions by ' + col)

    fig.tight_layout()
    fig.savefig(OUT / f"eda_{col}.png")
    plt.close(fig)


val_predictions = {}
rows = []
test_predictions = {}
fitted = {}


for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)

    pv = pipeline.predict_proba(X_val)[:, 1]

    val_predictions[name] = pv
    rows.append(scores(name, y_val, pv))
    fitted[name] = pipeline


for name, residual in [
    ('MLP', False),
    ('Residual Tabular NN', True)
]:
    pp, net, history = train_neural(residual=residual)

    pv = predict_neural(pp, net, X_val)

    val_predictions[name] = pv
    rows.append(scores(name, y_val, pv))
    fitted[name] = (pp, net)

    pd.DataFrame(
        history,
        columns=['epoch', 'train_loss', 'validation_loss']
    ).set_index('epoch').plot(
        title=name + ' learning curves',
        ylabel='BCE loss'
    ).get_figure().savefig(
        OUT / ('learning_' + name.replace(' ', '_') + '.png')
    )

    plt.close('all')


validation = pd.DataFrame(rows).sort_values(
    'macro_f1',
    ascending=False
)

validation.to_csv(OUT / 'validation_comparison.csv', index=False)


fig, ax = plt.subplots(figsize=(8, 4))
validation.set_index('model').macro_f1.plot.barh(ax=ax)

ax.set_xlabel('Validation macro-F1')
ax.set_title('Model comparison (validation only)')

fig.tight_layout()
fig.savefig(OUT / 'model_comparison.png')
plt.close(fig)


# Optimise validation winner, including neural case through a small dropout search.
winner = validation.iloc[0]['model']
best_name = winner

if winner in ['MLP', 'Residual Tabular NN']:
    candidates = []

    for drop in [0.1, 0.25, 0.4]:
        pp, net, h = train_neural(
            residual=(winner == 'Residual Tabular NN'),
            dropout=drop
        )

        p = predict_neural(pp, net, X_val)

        candidates.append((
            f1_score(y_val, p >= 0.5, average='macro'),
            drop,
            pp,
            net
        ))

    best = max(candidates, key=lambda t: t[0])
    champ = (best[2], best[3])

    tuning = {
        'method': 'validation dropout search',
        'dropout': best[1],
        'validation_macro_f1': best[0]
    }

else:
    if winner == 'Random Forest':
        grid = {
            'model__n_estimators': [100, 200],
            'model__min_samples_leaf': [2, 5, 10],
            'model__max_depth': [None, 12]
        }

    elif winner == 'Logistic Regression':
        grid = {
            'model__C': [0.1, 1, 10],
            'model__class_weight': [None, 'balanced']
        }

    else:
        grid = {
            'model__strategy': [
                'most_frequent',
                'stratified',
                'prior'
            ]
        }

    search = RandomizedSearchCV(
        models[winner],
        grid,
        n_iter=min(
            6,
            int(np.prod([len(v) for v in grid.values()]))
        ),
        scoring='f1_macro',
        cv=StratifiedKFold(
            3,
            shuffle=True,
            random_state=SEED
        ),
        random_state=SEED,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    champ = search.best_estimator_

    tuning = {
        'method': '3-fold randomized search',
        'parameters': search.best_params_,
        'cv_macro_f1': search.best_score_
    }


(OUT / 'tuning.json').write_text(
    json.dumps(tuning, indent=2)
)


# Test predictions only after all selection and tuning.
for name, est in fitted.items():
    if isinstance(est, tuple):
        p = predict_neural(*est, X_test)
    else:
        p = est.predict_proba(X_test)[:, 1]

    test_predictions[name] = p


if isinstance(champ, tuple):
    test_predictions['Optimized Champion'] = predict_neural(
        *champ,
        X_test
    )
else:
    test_predictions['Optimized Champion'] = (
        champ.predict_proba(X_test)[:, 1]
    )


rows = []
fig, ax = plt.subplots(figsize=(7, 6))


for name, p in test_predictions.items():
    rows.append(scores(name, y_test, p))

    pred = (p >= 0.5).astype(int)

    (
        OUT / ('classification_' + name.replace(' ', '_') + '.txt')
    ).write_text(
        classification_report(
            y_test,
            pred,
            target_names=['No interruption', 'Interruption']
        )
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix(y_test, pred),
        display_labels=['No', 'Yes']
    )

    disp.plot()

    disp.figure_.savefig(
        OUT / ('confusion_' + name.replace(' ', '_') + '.png')
    )

    plt.close(disp.figure_)

    RocCurveDisplay.from_predictions(
        y_test,
        p,
        name=name,
        ax=ax
    )


ax.set_title('ROC curves on shared held-out test set')
fig.tight_layout()
fig.savefig(OUT / 'roc_comparison.png')
plt.close(fig)


pd.DataFrame(rows).to_csv(
    OUT / 'test_comparison.csv',
    index=False
)


comparator = max(
    (n for n in val_predictions if n != winner),
    key=lambda n: f1_score(
        y_val,
        val_predictions[n] >= 0.5,
        average='macro'
    )
)

a = (
    (test_predictions['Optimized Champion'] >= 0.5)
    == y_test.to_numpy()
)

b = (
    (test_predictions[comparator] >= 0.5)
    == y_test.to_numpy()
)

n01 = int((a & ~b).sum())
n10 = int((~a & b).sum())

pval = (
    binomtest(
        min(n01, n10),
        n01 + n10,
        p=0.5
    ).pvalue
    if n01 + n10
    else 1.
)


(OUT / 'mcnemar.json').write_text(
    json.dumps({
        'comparator': comparator,
        'both_correct': int((a & b).sum()),
        'champion_only_correct': n01,
        'comparator_only_correct': n10,
        'both_wrong': int((~a & ~b).sum()),
        'exact_p_value': pval,
        'alpha': 0.05
    }, indent=2)
)


# Three controlled feature-group ablations; refit same champion family on fixed train/val.
abl = []

for removed in [
    ['prov', 'geotype'],
    ['wat_drinkwat'],
    ['hsg_maind', 'hsg_tenure']
]:
    cols = [c for c in FEATURES if c not in removed]

    if isinstance(champ, tuple):
        # Same neural architecture, changed feature input dimension; fixed validation protocol.
        pp = preprocessor(cols)

        tr = pp.fit_transform(
            X_train[cols]
        ).astype('float32')

        va = pp.transform(
            X_val[cols]
        ).astype('float32')

        net = NeuralNet(
            tr.shape[1],
            winner == 'Residual Tabular NN'
        )

        opt = torch.optim.AdamW(
            net.parameters(),
            lr=0.001
        )

        loss = nn.BCEWithLogitsLoss()

        for epoch in range(15):
            net.train()

            for ids in np.array_split(
                np.random.default_rng(SEED + epoch).permutation(len(tr)),
                max(1, int(np.ceil(len(tr) / 256)))
            ):
                opt.zero_grad()

                l = loss(
                    net(torch.tensor(tr[ids])),
                    torch.tensor(
                        y_train.to_numpy(dtype='float32')[ids]
                    )
                )

                l.backward()
                opt.step()

        p = predict_neural(pp, net, X_val[cols])

    else:
        from sklearn.base import clone

        est = clone(champ)
        est.set_params(pre=preprocessor(cols))
        est.fit(X_train[cols], y_train)

        p = est.predict_proba(X_val[cols])[:, 1]

    abl.append({
        'removed': '+'.join(removed),
        'validation_macro_f1': f1_score(
            y_val,
            p >= 0.5,
            average='macro'
        )
    })


pd.DataFrame(abl).to_csv(
    OUT / 'ablations.csv',
    index=False
)


# Export model and schema for UI; torch checkpoint when applicable.
meta = {
    'features': FEATURES,
    'champion': winner,
    'target': 'Reported water interruption in prior 12 months',
    'rows_eligible': len(df),
    'rows_test': len(X_test)
}


if isinstance(champ, tuple):
    pp, net = champ

    joblib.dump(
        pp,
        ROOT / 'models' / 'preprocessor.joblib'
    )

    torch.save({
        'state_dict': net.state_dict(),
        'input_dim': pp.transform(X_train.iloc[:1]).shape[1],
        'residual': winner == 'Residual Tabular NN'
    }, ROOT / 'models' / 'neural.pt')

    meta['type'] = 'torch'

else:
    joblib.dump(
        champ,
        ROOT / 'models' / 'pipeline.joblib'
    )

    meta['type'] = 'sklearn'


(ROOT / 'models' / 'metadata.json').write_text(
    json.dumps(meta, indent=2)
)


(OUT / 'dataset_summary.json').write_text(
    json.dumps({
        'original_rows': int(
            pd.read_csv(DATA, usecols=[TARGET]).shape[0]
        ),
        'eligible_rows': len(df),
        'positive': int(y.sum()),
        'negative': int((1 - y).sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'features': FEATURES
    }, indent=2)
)


print('Finished. Champion:', winner, '; outputs:', OUT)
