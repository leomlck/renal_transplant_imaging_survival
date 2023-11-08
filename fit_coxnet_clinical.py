import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import re
import json
import random

import warnings
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

import sksurv.util
from sksurv.linear_model import CoxnetSurvivalAnalysis

from utils_survival_analysis import *

import argparse

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FitFailedWarning)
warnings.simplefilter("ignore", ArithmeticError)
warnings.simplefilter("ignore", ConvergenceWarning)

random.seed(88)
np.random.seed(88)

parser = argparse.ArgumentParser(description='features ml')
parser.add_argument('--event', type=str,
                    help="event; acute(ar) or chronic(cr)")
parser.add_argument('--test', default=0, type=int)
args = parser.parse_args()
print('\n', args)

n_jobs = 16
#n_jobs = int(os.cpu_count()/n_jobs)
#print('Number of cpus:',os.cpu_count())
print('Number of jobs:', n_jobs)
years = 5 if args.event=='cr' else 2 

main_path = '/gpfs/users/mileckil/kidney_workspace/project_kidney/survival_workspace/save'
features_path = '/gpfs/workdir/mileckil/data'
data_path = '/gpfs/workdir/mileckil/data/clinicobiological_data'

df_features = pd.read_csv(os.path.join(features_path,'dataframes/df_clinical_variables.csv'), sep=';', index_col=0)
df_features = df_features.drop(columns=['nb_artere_tot', 'nb_artere_anast', 'anast_urin',
       'anast_arter_patch', 'anast_arter_type', 'anast_veine'])
df_features = df_features.dropna()
if args.event == 'ar':
    df_features = df_features.drop(columns=['complic_nbr'])
print(df_features.columns)

feat_dim = len(df_features.columns)-1
print('Features dimension : ', feat_dim)
print('Number of patient features : ', len(df_features))
#df_features
df_features = df_features.loc[:, (df_features != 0).any(axis=0)]

save_folder = 'results_survival_analysis_{}_ext'.format(args.event)
save_folder = os.path.join(main_path, save_folder, 'clinical', 'CoxnetSurvivalAnalysis')
os.makedirs(save_folder, exist_ok=True)

if args.event == 'cr':
    df_cr = get_df_cr(main_path, data_path, df_features, years)
elif args.event == 'ar':
    df_cr = get_df_ar(main_path, data_path, df_features, years)
else:
    print('Invalid event arg')

n_splits = 3
if args.test:
    n_alphas = 5 
    l1_ratios = [0.1, 1]
    thresholds_ids = [16, 128]
else:
    n_splits = 3
    n_alphas = 100
    l1_ratios = np.linspace(0.1, 1, 10)
    thresholds_ids = [16, 32, 64, 128, 256]
scores = []

# cox

# imaging
print('\n### Imaging features')
X = df_features.merge(df_cr[['patient', 'event', 'duration']], how='left', on='patient')
y = sksurv.util.Surv.from_dataframe('event', 'duration', X)
X = X.drop(columns=['event', 'duration', 'patient'])

coxnet_pipe = make_pipeline(
    StandardScaler(),
    CoxnetSurvivalAnalysis(n_alphas=n_alphas, l1_ratio=0.5, alpha_min_ratio=0.01, max_iter=1000)
)
coxnet_pipe.fit(X, y)

estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
#estimated_alphas = np.logspace(-3, 3, 7)
cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
gcv = GridSearchCV(
    make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(max_iter=500000)),
    param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas],
                "coxnetsurvivalanalysis__l1_ratio": l1_ratios},
    cv=cv,
    error_score=0.5,
    refit=False,
    n_jobs=n_jobs,
    verbose=1).fit(X, y)

cv_results = pd.DataFrame(gcv.cv_results_)
cv_results['param_coxnetsurvivalanalysis__alphas'] = cv_results['param_coxnetsurvivalanalysis__alphas'].apply(lambda x: float(x[0]))

fig = plot_cv_results(cv_results, 'coxnetsurvivalanalysis__alphas', 'coxnetsurvivalanalysis__l1_ratio')
fig.savefig(os.path.join(main_path, save_folder, 'coxnet_cv_results'), bbox_inches='tight')
plt.close()

coxnet_pred = make_pipeline(
    StandardScaler(),
    CoxnetSurvivalAnalysis(max_iter=500000, fit_baseline_model=True)
)
coxnet_pred.set_params(**gcv.best_params_)
try:
    coxnet_pred.fit(X, y)
    train_score = coxnet_pred.score(X, y)
except ArithmeticError:
    train_score = .5
    test_score = .5

best_coefs = pd.DataFrame(
    coxnet_pred.named_steps["coxnetsurvivalanalysis"].coef_,
    index=X.columns,
    columns=["coefficient"]
)

non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
print("Number of non-zero coefficients: {}".format(non_zero))

non_zero_coefs = best_coefs.query("coefficient != 0")
coef_order = non_zero_coefs.abs().sort_values("coefficient").index

fig, ax = plt.subplots(figsize=(6, 18))
non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
ax.set_xlabel("coefficient")
ax.grid(True)
plt.gcf().patch.set_facecolor('white')
fig.savefig(os.path.join(main_path, save_folder, 'best_coxnet_coefs'), bbox_inches='tight')
plt.close()

try:
    train_crossval_scores = cross_val_score(coxnet_pred, X, y, cv=cv, n_jobs=n_jobs)
    mean_train_crossval_scores, std_train_crossval_scores = np.mean(train_crossval_scores), np.std(train_crossval_scores)
except ValueError:
    mean_train_crossval_scores, std_train_crossval_scores = 0.5, 0.

scores.append([train_score, mean_train_crossval_scores, std_train_crossval_scores])

ordered_feats = non_zero_coefs.abs().sort_values("coefficient")
s_feats = {'best_model_ordered_feats': ordered_feats.index.tolist()}
with open(os.path.join(main_path, save_folder, 'ordered_feats_coxnet.json'), 'w') as fp:
    json.dump(s_feats, fp)

for thresholds_id in thresholds_ids:
    selected_feats = ordered_feats.head(thresholds_id).index.tolist()
    Xs = X[selected_feats]

    coxnet_pipe = make_pipeline(
        StandardScaler(),
        CoxnetSurvivalAnalysis(n_alphas=n_alphas, l1_ratio=0.5, alpha_min_ratio=0.01, max_iter=100)
    )
    coxnet_pipe.fit(Xs, y)

    estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    gcv = GridSearchCV(
        make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(max_iter=500000)),
        param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas],
                    "coxnetsurvivalanalysis__l1_ratio": l1_ratios},
        cv=cv,
        error_score=0.5,
        refit=False,
        n_jobs=n_jobs,
        verbose=1).fit(Xs, y)

    cv_results = pd.DataFrame(gcv.cv_results_)
    cv_results['param_coxnetsurvivalanalysis__alphas'] = cv_results['param_coxnetsurvivalanalysis__alphas'].apply(lambda x: float(x[0]))

    fig = plot_cv_results(cv_results, 'coxnetsurvivalanalysis__alphas', 'coxnetsurvivalanalysis__l1_ratio')
    fig.savefig(os.path.join(main_path, save_folder, 'coxnet_cv_results_selected_{}'.format(thresholds_id)), bbox_inches='tight')
    plt.close()

    coxnet_pred = make_pipeline(
        StandardScaler(),
        CoxnetSurvivalAnalysis(max_iter=500000, fit_baseline_model=True)
    )
    coxnet_pred.set_params(**gcv.best_params_)
    try:
        coxnet_pred.fit(X, y)
        train_score = coxnet_pred.score(X, y)
    except ArithmeticError:
        train_score = .5
        test_score = .5
    
    try:
        train_crossval_scores = cross_val_score(coxnet_pred, X, y, cv=cv, n_jobs=n_jobs)
        mean_train_crossval_scores, std_train_crossval_scores = np.mean(train_crossval_scores), np.std(train_crossval_scores)
    except ValueError:
        mean_train_crossval_scores, std_train_crossval_scores = 0.5, 0.

    scores.append([train_score, mean_train_crossval_scores, std_train_crossval_scores])


df_scores = pd.DataFrame(scores, columns=['train', 'mean_3CV_train', 'std_3CV_train'], index=['all']+['selected_{}'.format(thresholds_id) for thresholds_id in thresholds_ids])
df_scores.to_csv(os.path.join(main_path, save_folder, 'scores_coxnet.csv'))



