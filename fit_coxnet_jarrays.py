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

parser = argparse.ArgumentParser(description='features ml')
parser.add_argument('--settings', default='dce_J15_6', type=str)
parser.add_argument('--test', default=0, type=int)
parser.add_argument('--job_index', default=0, type=int)
args = parser.parse_args()

df_params = pd.read_csv('configs/df_args_settings_{}.csv'.format(args.settings))
params = df_params.iloc[args.job_index-1]
args.event = params['event']
args.exam = params['exam']
args.mod = params['mod']
args.model = params['model']
args.pretraining = params['pretraining']
args.clinical = params['clinical']
args.seed = 8 #int(params['seed'])
print('\n', args)

random.seed(args.seed)
np.random.seed(args.seed)

n_jobs = 32
#n_jobs = int(os.cpu_count()/n_jobs)
#print('Number of cpus:',os.cpu_count())
print('Number of jobs:', n_jobs)
years = 5 if args.event=='cr' else 2 

main_path = '/gpfs/users/mileckil/kidney_workspace/project_kidney/survival_workspace/save'
features_path = '/gpfs/workdir/mileckil/output'
data_path = '/gpfs/workdir/mileckil/data/clinicobiological_data'
dataframes_path = '/gpfs/workdir/mileckil/data/dataframes'
index_col = None if args.model=='radiomics'  else 0

path_to_features = os.path.join(features_path, 'save_features/save_features_{}_{}/features_{}_{}.csv'.format(args.model, args.mod, args.exam, args.pretraining))
df_features = pd.read_csv(path_to_features, index_col=index_col)
if index_col == 0:
    df_features = df_features.filter(regex=re.compile(r'(\d+|patient)'), axis=1)

feat_dim = len(df_features.columns)-1
print('Features dimension : ', feat_dim)
print('Number of patient features : ', len(df_features))
#df_features
df_features = df_features.loc[:, (df_features != 0).any(axis=0)]

if args.clinical:
    df_clinical = pd.read_csv(os.path.join(dataframes_path, 'df_clinical_variables_fill.csv'), index_col=0)
    df_features = df_features.merge(df_clinical, how='left', on='patient') #df_features[['patient']]
    
save_folder = 'results_survival_analysis_{}_ext'.format(args.event)
save_folder = os.path.join(main_path, save_folder, args.mod, args.pretraining, args.exam, 'CoxnetSurvivalAnalysis_el')
if args.clinical:
    save_folder += '_w_clinical'
os.makedirs(save_folder, exist_ok=True)

if args.event == 'cr':
    df_cr = get_df_cr(main_path, data_path, df_features, years)
elif args.event == 'ar':
    df_cr = get_df_ar(main_path, data_path, df_features, years)
else:
    print('Invalid event arg')

# import external data
path_to_features_ext = os.path.join(features_path, 'save_features/save_features_{}_external_{}/features_external_{}.csv'.format(args.model, args.mod, args.pretraining))
df_features_ext = pd.read_csv(path_to_features_ext, index_col=index_col)
if index_col == 0:
    df_features_ext = df_features_ext.filter(regex=re.compile(r'(\d+|patient|exam)'), axis=1)

df_features_ext = df_features_ext.loc[:, (df_features_ext != 0).any(axis=0)]
df_features_ext = df_features_ext[df_features_ext.exam <= 30]
if args.clinical:
    df_clinical_ext = pd.read_csv(os.path.join(dataframes_path, 'df_clinical_variables_ext_fill.csv'), index_col=0)
    df_features_ext = df_features_ext.merge(df_clinical_ext, how='left', on='patient') #df_features_ext[['patient', 'exam']]
 
df_ext = prepare_ext_df(main_path)

df_ext['event'] = df_ext['{}'.format(args.event)]
df_ext['duration'] = pd.Timedelta(years*365, 'D')
if args.event =='cr':
    df_ext['duration'].loc[df_ext['cr']==1] = df_ext['cr date'] - df_ext['transplantation date']
    df_ext['duration'].loc[df_ext['cr']==0] = pd.to_datetime('01-01-2023') - df_ext['transplantation date']
    df_ext['duration'] = df_ext['duration'].apply(lambda x: x.total_seconds()/86400)
    df_ext['duration'].loc[df_ext['duration']>years*365] = years*365
elif args.event == 'ar':
    df_ext['duration'].loc[df_ext['ar']==1] = df_ext['ar date 1'] - df_ext['transplantation date']
    df_ext['duration'] = df_ext['duration'].apply(lambda x: x.total_seconds()/86400)

n_splits = 3
if args.test:
    n_alphas = 5
    l1_ratios = [0.1, 1]
    thresholds_ids = [512]
else:
    n_alphas = 200
    l1_ratios = np.linspace(0.05, 1, 20)
    thresholds_ids = [512, 16, 32, 64] #[4, 8, 16]
scores = []

# cox

# imaging
print('\n### Imaging features')
X = df_features.merge(df_cr[['patient', 'event', 'duration']], how='left', on='patient')
y = sksurv.util.Surv.from_dataframe('event', 'duration', X)
X = X.drop(columns=['event', 'duration', 'patient'])

Xt = df_features_ext.merge(df_ext[['patient', 'event', 'duration']], how='left', on='patient')
Xt_early = Xt.sort_values(by='exam').drop_duplicates(subset='patient', keep='first')
Xt_late = Xt.sort_values(by='exam').drop_duplicates(subset='patient', keep='last')
print('len(Xt_early) :', len(Xt_early))
print("Xt_early['event'].sum() :", Xt_early['event'].sum())
print('len(Xt_late) :', len(Xt_late))
print("Xt_late['event'].sum() :", Xt_late['event'].sum())
yt_early = sksurv.util.Surv.from_dataframe('event', 'duration', Xt_early)
yt_late = sksurv.util.Surv.from_dataframe('event', 'duration', Xt_late)
Xt_early = Xt_early.drop(columns=['event', 'duration', 'patient', 'exam'])
Xt_late = Xt_late.drop(columns=['event', 'duration', 'patient', 'exam'])

coxnet_pipe = make_pipeline(
    StandardScaler(),
    CoxnetSurvivalAnalysis(n_alphas=n_alphas, l1_ratio=0.5, alpha_min_ratio=0.01, max_iter=100000)
)
coxnet_pipe.fit(X, y)

estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
#estimated_alphas = np.logspace(-3, 3, 7)
cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
gcv = GridSearchCV(
    make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(max_iter=100000)),
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
    CoxnetSurvivalAnalysis(max_iter=100000, fit_baseline_model=True)
)
coxnet_pred.set_params(**gcv.best_params_)
try:
    coxnet_pred.fit(X, y)
    train_score = coxnet_pred.score(X, y)
    test_score_early = coxnet_pred.score(Xt_early, yt_early)
    test_score_late = coxnet_pred.score(Xt_late, yt_late)
except ArithmeticError:
    train_score = .5
    test_score_early = .5
    test_score_late = .5

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
    coxnet_pred.fit(Xt_early, yt_early)
    trained_test_score_early = coxnet_pred.score(Xt_early, yt_early)
    main_test_score_early = coxnet_pred.score(X, y)
except ArithmeticError:
    trained_test_score_early = .5
    main_test_score_early = .5

try:
    coxnet_pred.fit(Xt_late, yt_late)
    trained_test_score_late = coxnet_pred.score(Xt_late, yt_late)
    main_test_score_late = coxnet_pred.score(X, y)
except ArithmeticError:
    trained_test_score_late = .5
    main_test_score_late = .5
'''
try:
    train_crossval_scores = cross_val_score(coxnet_pred, X, y, cv=cv, n_jobs=n_jobs)
    mean_train_crossval_scores, std_train_crossval_scores = np.mean(train_crossval_scores), np.std(train_crossval_scores)
except ValueError:
    mean_train_crossval_scores, std_train_crossval_scores = 0.5, 0.
'''
mean_train_crossval_scores_main = cv_results['mean_test_score'][gcv.best_index_]
std_train_crossval_scores_main = cv_results['std_test_score'][gcv.best_index_]

try:
    test_crossval3_scores_early = cross_val_score(coxnet_pred, Xt_early, yt_early, cv=cv, n_jobs=n_jobs)
    mean_test_crossval3_scores_early, std_test_crossval3_scores_early = np.mean(test_crossval3_scores_early), np.std(test_crossval3_scores_early)
except ValueError:
    mean_test_crossval3_scores_early, std_test_crossval3_scores_early = 0.5, 0.

try:
    test_crossval3_scores_late = cross_val_score(coxnet_pred, Xt_late, yt_late, cv=cv, n_jobs=n_jobs)
    mean_test_crossval3_scores_late, std_test_crossval3_scores_late = np.mean(test_crossval3_scores_late), np.std(test_crossval3_scores_late)
except ValueError:
    mean_test_crossval3_scores_late, std_test_crossval3_scores_late = 0.5, 0.

scores.append([train_score, test_score_early, test_score_late,
               trained_test_score_early, trained_test_score_late,
               main_test_score_early, trained_test_score_late,
               mean_train_crossval_scores_main, std_train_crossval_scores_main,
               0, 0,
               mean_test_crossval3_scores_early, std_test_crossval3_scores_early,
               mean_test_crossval3_scores_late, std_test_crossval3_scores_late])

ordered_feats = non_zero_coefs.abs().sort_values("coefficient", ascending=False)
s_feats = {'best_model_ordered_feats': ordered_feats.index.tolist()}
with open(os.path.join(main_path, save_folder, 'ordered_feats_coxnet.json'), 'w') as fp:
    json.dump(s_feats, fp)

for thresholds_id in thresholds_ids:
    selected_feats = ordered_feats.head(thresholds_id).index.tolist()
    Xs = X[selected_feats]

    coxnet_pipe = make_pipeline(
        StandardScaler(),
        CoxnetSurvivalAnalysis(n_alphas=n_alphas, l1_ratio=0.5, alpha_min_ratio=0.01, max_iter=100000)
    )
    coxnet_pipe.fit(Xs, y)

    estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    gcv = GridSearchCV(
        make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(max_iter=100000)),
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
        CoxnetSurvivalAnalysis(max_iter=100000, fit_baseline_model=True)
    )
    coxnet_pred.set_params(**gcv.best_params_)
    
    try:
        coxnet_pred.fit(X, y)
        train_score = coxnet_pred.score(X, y)
        test_score_early = coxnet_pred.score(Xt_early, yt_early)
        test_score_late = coxnet_pred.score(Xt_late, yt_late)
    except ArithmeticError:
        train_score = .5
        test_score_early = .5
        test_score_late = .5

    try:
        coxnet_pred.fit(Xt_early, yt_early)
        trained_test_score_early = coxnet_pred.score(Xt_early, yt_early)
        main_test_score_early = coxnet_pred.score(X, y)
    except ArithmeticError:
        trained_test_score_early = .5
        main_test_score_early = .5

    try:
        coxnet_pred.fit(Xt_late, yt_late)
        trained_test_score_late = coxnet_pred.score(Xt_late, yt_late)
        main_test_score_late = coxnet_pred.score(X, y)
    except ArithmeticError:
        trained_test_score_late = .5
        main_test_score_late = .5
    '''
    try:
        train_crossval_scores = cross_val_score(coxnet_pred, X, y, cv=cv, n_jobs=n_jobs)
        mean_train_crossval_scores, std_train_crossval_scores = np.mean(train_crossval_scores), np.std(train_crossval_scores)
    except ValueError:
        mean_train_crossval_scores, std_train_crossval_scores = 0.5, 0.
    '''
    mean_train_crossval_scores = cv_results['mean_test_score'][gcv.best_index_]
    std_train_crossval_scores = cv_results['std_test_score'][gcv.best_index_]


    try:
        test_crossval3_scores_early = cross_val_score(coxnet_pred, Xt_early, yt_early, cv=cv, n_jobs=n_jobs)
        mean_test_crossval3_scores_early, std_test_crossval3_scores_early = np.mean(test_crossval3_scores_early), np.std(test_crossval3_scores_early)
    except ValueError:
        mean_test_crossval3_scores_early, std_test_crossval3_scores_early = 0.5, 0.

    try:
        test_crossval3_scores_late = cross_val_score(coxnet_pred, Xt_late, yt_late, cv=cv, n_jobs=n_jobs)
        mean_test_crossval3_scores_late, std_test_crossval3_scores_late = np.mean(test_crossval3_scores_late), np.std(test_crossval3_scores_late)
    except ValueError:
        mean_test_crossval3_scores_late, std_test_crossval3_scores_late = 0.5, 0.

    scores.append([train_score, test_score_early, test_score_late,
                   trained_test_score_early, trained_test_score_late,
                   main_test_score_early, main_test_score_late, 
                   mean_train_crossval_scores_main, std_train_crossval_scores_main,
                   mean_train_crossval_scores, std_train_crossval_scores,
                   mean_test_crossval3_scores_early, std_test_crossval3_scores_early,
                   mean_test_crossval3_scores_late, std_test_crossval3_scores_late])

df_scores = pd.DataFrame(scores, columns=['train', 'test_early', 'test_late',
                                          'trained_test_early', 'trained_test_late',
                                          'main_early', 'main_late', 
                                          'mean_3CV_train_main', 'std_3CV_train_main',
                                          'mean_3CV_train', 'std_3CV_train',
                                          'mean_3CV_test_early', 'std_3CV_test_early',
                                          'mean_3CV_test_late', 'std_3CV_test_late'], index=['all']+['selected_{}'.format(thresholds_id) for thresholds_id in thresholds_ids])
df_scores.to_csv(os.path.join(main_path, save_folder, 'scores_coxnet.csv'))



