import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import re
import json

from utils_survival_analysis import *

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.cluster import KMeans

from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter

import argparse

parser = argparse.ArgumentParser(description='features ml')
parser.add_argument('--model', type=str,
                    help="model")
parser.add_argument('--mod', type=str,
                    help="modality")
parser.add_argument('--pretraining', type=str,
                    help="pretraining")
parser.add_argument('--event', type=str,
                    help="event; acute(ar) or chronic(cr)")
parser.add_argument('--exam', type=str,
                    help="exam")
parser.add_argument('--survival_model', choices=['rsf', 'coxnet'], type=str,
                    help="survival model for selected features")
args = parser.parse_args()
print('\n', args)

n_jobs = 4 
#n_jobs = int(os.cpu_count()/n_jobs)
#print('Number of cpus:',os.cpu_count())
#print('Number of jobs:', n_jobs)
years = 5 if args.event=='cr' else 2 
normalize = True

# Data, features, model paths

main_path = '/gpfs/users/mileckil/kidney_workspace/project_kidney/survival_workspace/save'
features_path = '/gpfs/workdir/mileckil/output'
data_path = '/gpfs/workdir/mileckil/data/clinicobiological_data'
model_folder = 'RandomSurvivalForest_el' if args.survival_model=='rsf' else 'CoxnetSurvivalAnalysis_el'

# Load features

path_to_features = os.path.join(features_path, 'save_features/save_features_{}_{}/features_{}_{}.csv'.format(args.model, args.mod, args.exam, args.pretraining))
df_features = pd.read_csv(path_to_features, index_col=0)
df_features = df_features.filter(regex=re.compile(r'(\d+|patient)'), axis=1)
if normalize:
    scaler = make_pipeline(StandardScaler())
    df_n = df_features.drop(columns=['patient'])
    scaler.fit(df_n)
    scaled_data = scaler.transform(df_n)
    df_n = pd.DataFrame(scaled_data, columns=df_n.columns, index=df_n.index)
    df_features = pd.concat([df_n, df_features[['patient']]], axis=1)

feat_dim = len(df_features.columns)-1
print('Features dimension : ', feat_dim)
print('Number of patient features : ', len(df_features))
#df_features
df_features = df_features.loc[:, (df_features != 0).any(axis=0)]

save_folder = 'results_survival_analysis_{}_ext'.format(args.event)
save_folder = os.path.join(main_path, save_folder, args.mod, args.pretraining, args.exam, model_folder)

# Load selected features by the survival model

with open(os.path.join(main_path, save_folder, 'ordered_feats_{}.json'.format(args.survival_model))) as fp:
    selected_feats = json.load(fp)
selected_feats = selected_feats['best_model_ordered_feats']

if args.event == 'cr':
    df_cr = get_df_cr(main_path, data_path, df_features, years)
elif args.event == 'ar':
    df_cr = get_df_ar(main_path, data_path, df_features, years)
else:
    print('Invalid event arg')

# Clustering features

n_clusters = 2

dictionnary_clusterers = {
    "KMeans" : KMeans(n_clusters=n_clusters)
}

X = df_features.merge(df_cr[['patient', 'event', 'duration']], how='left', on='patient').drop(columns='patient')
X = X[['event', 'duration']+selected_feats]

clusterers_labels = np.empty((len(X), len(dictionnary_clusterers.keys())))
for i, clusterer_name in enumerate(dictionnary_clusterers.keys()):
    t = time.time()
    clusterer = dictionnary_clusterers[clusterer_name]
    clusterer.fit(X.drop(columns=['event', 'duration']))
    clusterers_labels[:,i] = clusterer.labels_
    print('Time to cluster with {} : {} min.'.format(clusterer_name, (time.time()-t)/60))

columns = [x+'_labels' for x in dictionnary_clusterers.keys()]
df_clusterers_labels = pd.DataFrame(clusterers_labels, columns=columns, index=X.index)
XX = pd.concat((X, df_clusterers_labels), axis=1)

dict_targets = {}
dict_targets['clusters'] = columns
dict_target_thresholds = {}
dict_target_thresholds['clusters'] = [0 for i in range(len(columns))]

group = 'clusters'
targets = dict_targets[group]
target_thresholds = dict_target_thresholds[group]
for target in targets:
    if (XX[target].sum() == 0) or (XX[target].sum() == len(XX)):
        targets.remove(target)
target_thresholds = dict_target_thresholds[group]

T = XX["duration"]
E = XX["event"]

# Plot Kaplan-Meier curves

kmf = KaplanMeierFitter()
fig, ax = plt.subplots(len(targets), 1, figsize=(5,5*len(targets))) 
for i, target in enumerate(targets):
    target_threshold = target_thresholds[i]
    print((target, target_threshold))
    mask = XX[target] > target_threshold
    kmf.fit(T[mask], event_observed=E[mask], label="{} target > {:.2f}".format(target, target_threshold))
    kmf.plot_survival_function(ax=ax[i])

    mask = XX[target] <= target_threshold
    kmf.fit(T[mask], event_observed=E[mask], label="{} target <= {:.2f}".format(target, target_threshold))
    kmf.plot_survival_function(ax=ax[i])

    ax[i].set_ylabel("est. probability of survival")
    ax[i].set_xlabel("time $t$ (days)")
    ax[i].legend(loc="best")
    ax[i].set_xlim((0,years*365))
    ax[i].set_ylim((0.,1.05))

    results = logrank_test(T[mask], T[~mask], E[mask], E[~mask], alpha=.99)
    ax[i].set_title('test_statistic: {:.2f} | p: {:.4f} | -log2(p): {:.4f}'.format(results.summary['test_statistic'].values[0], 
                                                                       results.summary['p'].values[0], 
                                                                       results.summary['-log2(p)'].values[0]))
plt.tight_layout()
fig.savefig(os.path.join(main_path, save_folder, 'kaplan_meier_imaging_selected_{}'.format(args.survival_model)))


