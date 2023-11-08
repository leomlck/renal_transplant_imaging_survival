import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from sklearn.model_selection import StratifiedKFold

from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter

def prepare_ext_df(main_path):
    df_ext = pd.read_csv(os.path.join(main_path,'liste_irm_greffons_externe_FP_leo.csv'))
    df_ext['death date'] = pd.to_datetime(df_ext['death date'], format='%d/%m/%Y')
    df_ext['transplantation date'] = pd.to_datetime(df_ext['transplantation date'], format='%d/%m/%Y')
    df_ext['cr date'].loc[df_ext['cr date']=='Non'] = np.nan
    df_ext['cr date'] = pd.to_datetime(df_ext['cr date'], format='%m/%d/%Y')
    df_ext['ar date 1'] = pd.to_datetime(df_ext['ar date 1'], format='%m/%d/%Y')
    df_ext['ar date 2'] = pd.to_datetime(df_ext['ar date 2'], format='%m/%d/%Y')
    df_ext['ar date 3'] = pd.to_datetime(df_ext['ar date 3'], format='%m/%d/%Y')
    df_ext['cr'] = df_ext['cr'].apply(lambda x: 1 if x=='Oui' else 0 if x=='Non' else np.nan)
    df_ext['ar'] = df_ext['ar'].apply(lambda x: 1 if x=='Oui' else 0 if x=='Non' else np.nan)
    return df_ext

def prepare_complication_df(path):    
    df=pd.read_csv(os.path.join(path, 'imag-nct_export-csv_20190722_FGN/2_COMPLICATION.csv'), sep=';')
    df['ddeb_complic'] = pd.to_datetime(df['ddeb_complic'], format='%d/%m/%Y')
    df['dfin_complic'] = pd.to_datetime(df['dfin_complic'], format='%d/%m/%Y')
    return df

def prepare_suivi_df(path):    
    df=pd.read_csv(os.path.join(path, 'imag-nct_export-csv_20190722_FGN/3_SUIVI.csv'), sep=';')
    df['dte_exam'] = pd.to_datetime(df['dte_exam'], format='%d/%m/%Y')
    df['biopsie_dte'] = pd.to_datetime(df['biopsie_dte'], format='%d/%m/%Y')
    return df

def prepare_inclusion_df(path):    
    df=pd.read_csv(os.path.join(path, 'imag-nct_export-csv_20190722_FGN/1_INCLUSION.csv'), sep=';')
    df['dte_incl'] = pd.to_datetime(df['dte_incl'], format='%d/%m/%Y')
    df['ddn'] = pd.to_datetime(df['ddn'], format='%d/%m/%Y')
    df['dte_transplant'] = pd.to_datetime(df['dte_transplant'], format='%d/%m/%Y')
    df['age_transplant'] = df['dte_transplant'] - df['ddn']
    df['age_transplant'] = df['age_transplant'].apply(lambda x: x.total_seconds()/(60*60*24*365.25))
    df['taille'] = df['taille'].apply(lambda x: x/100 if not np.isnan(x) and x>100 else x)
    df[['complic_chirtypec1', 
        'complic_chirtypec2', 
        'complic_chirtypec3', 
        'complic_chirtypec4', 
        'complic_chirtypec5', 
        'complic_chirtypec6', 
        'complic_chirtypec7']] = df[['complic_chirtypec1',
                                     'complic_chirtypec2', 
                                     'complic_chirtypec3', 
                                     'complic_chirtypec4', 
                                     'complic_chirtypec5', 
                                     'complic_chirtypec6', 
                                     'complic_chirtypec7']].fillna(value=0)
    return df

def prepare_last_followup_df(path):
    df = pd.read_csv(os.path.join(path, 'IMAGNCT_Last_followup_july2019.csv'), sep=';')
    df = df.drop(columns=['Unnamed: 7'])
    df.columns = ['DIVAT', 'Date of birth', 'Date of transplantation', 'Failure transplantation','Date return dialysis', 'Death', 'Date of death']

    df['Date of birth'] = pd.to_datetime(df['Date of birth'], format='%d/%m/%Y')
    df['Date of transplantation'] = pd.to_datetime(df['Date of transplantation'], format='%d/%m/%Y')
    df['Date return dialysis'] = pd.to_datetime(df['Date return dialysis'], format='%d/%m/%Y')
    df['Date of death'] = pd.to_datetime(df['Date of death'], format='%d/%m/%Y')
    df['Failure transplantation'] = df['Failure transplantation'].apply(lambda x: True if x[:3] == 'Oui' else (False if x == 'Non' else np.nan))
    df['Death'] = df['Death'].apply(lambda x: True if x == 'O' else (False if x == 'N' else np.nan))
    df = df[df['Failure transplantation'].notna()]
    return df

def get_df_cr(main_path, data_path, df_features, year):
    # clinical variables
    df_clinical = pd.read_csv(os.path.join(main_path,'df_clinical_variables.csv'), sep=';', index_col=0)
    df_clinical = df_clinical[['patient', 'sex', 'bmi', 'type_gref', 'nephropathie', 'cote_gref', 'age_transplant', 'delay_dte_incl']]

    # clinical target variables
    df_targets = pd.read_csv(os.path.join(main_path,'df_targets.csv'), index_col=0)
    df_targets = df_targets[['patient', 'nb_gref_prec', 'age_donneur', 'incomp_gref', 'dur_ischem_froide_m', 'complic_chir', 'mdrd J15', 'mdrd J30', 'mdrd M3', 'mdrd M12']]
    df_targets = df_targets.merge(df_clinical, how='left', on='patient')

    df_complication = prepare_complication_df(data_path)

    df_suivi = prepare_suivi_df(data_path)

    df_inclusion = prepare_inclusion_df(data_path)
    df_inclusion = df_inclusion.dropna(subset=['num_divat'])
    df_inclusion['DIVAT'] = df_inclusion['num_divat'].astype(int)

    df_lastfollowup = prepare_last_followup_df(data_path)
    df_lastfollowup = df_lastfollowup.merge(df_inclusion[['patient', 'DIVAT']], on='DIVAT')

    df_cr = df_targets.merge(df_lastfollowup[['patient', 'Date of transplantation', 'Date return dialysis', 
                          'Date of death']], how='left', on='patient')
    df_cr = df_features.merge(df_cr, how='left', on='patient')
    df_cr['event'] = False
    df_cr['duration'] = pd.Timedelta(year*365, 'D')
    #df_cr['duration'].loc[~df_cr['Date of death'].isna()] = df_cr['Date of death'] - df_cr['Date of transplantation']
    #df_cr['event'].loc[~df_cr['Date of death'].isna()] = True
    df_cr['duration'].loc[~df_cr['Date return dialysis'].isna()] = df_cr['Date return dialysis'] - df_cr['Date of transplantation']
    df_cr['event'].loc[~df_cr['Date return dialysis'].isna()] = True

    df_cr['duration'] = df_cr['duration'].apply(lambda x: x.total_seconds()/86400)
    df_cr['event'] = df_cr['event'].astype(int)

    df_cr.dropna(subset=['duration'], inplace=True)
    return df_cr

def get_df_ar(main_path, data_path, df_features, years):
    # clinical variables
    df_clinical = pd.read_csv(os.path.join(main_path,'df_clinical_variables.csv'), sep=';', index_col=0)
    df_clinical = df_clinical[['patient', 'sex', 'bmi', 'type_gref', 'nephropathie', 'cote_gref', 'age_transplant', 'delay_dte_incl']]

    # clinical target variables
    df_targets = pd.read_csv(os.path.join(main_path,'df_targets.csv'), index_col=0)
    df_targets = df_targets[['patient', 'nb_gref_prec', 'age_donneur', 'incomp_gref', 'dur_ischem_froide_m', 'complic_chir', 'mdrd J15', 'mdrd J30', 'mdrd M3', 'mdrd M12']]
    df_targets = df_targets.merge(df_clinical, how='left', on='patient')

    df_complication = prepare_complication_df(data_path)

    df_suivi = prepare_suivi_df(data_path)

    df_inclusion = prepare_inclusion_df(data_path)
    df_inclusion = df_inclusion.dropna(subset=['num_divat'])
    df_inclusion['DIVAT'] = df_inclusion['num_divat'].astype(int)

    df_lastfollowup = prepare_last_followup_df(data_path)
    df_lastfollowup = df_lastfollowup.merge(df_inclusion[['patient', 'DIVAT']], on='DIVAT')

    df_comp = df_complication[['patient', 'type_complic', 'ddeb_complic']]
    df_comp = df_comp[df_comp.type_complic.isin([2,3])] #[1,2,3,4,7,8] or [2,3]
    print(len(df_comp))
    # remove patients with moree than 1 event
    for patient in np.unique(df_comp['patient'].values):
        #print((patient, len(df_ar[df_ar['patient']==patient])))
        if len(df_comp[df_comp['patient']==patient]) > 1:
            #display(df_ar[df_ar['patient']==patient].sort_values(by='ddeb_complic'))
            to_remove = df_comp[df_comp['patient']==patient].sort_values(by='ddeb_complic').index[1:]
            df_comp = df_comp.drop(index=to_remove)

    df_ar = df_targets.merge(df_lastfollowup[['patient', 'Date of transplantation']], how='left', on='patient')
    df_ar = df_ar.merge(df_comp, how='left', on='patient')
    df_ar = df_features.merge(df_ar, how='left', on='patient')
    df_ar['event'] = False
    df_ar['duration'] = pd.Timedelta(years*365, 'D')


    df_ar['duration'].loc[~df_ar['ddeb_complic'].isna()] = df_ar['ddeb_complic'] - df_ar['Date of transplantation']
    df_ar['event'].loc[~df_ar['ddeb_complic'].isna()] = True

    df_ar['duration'] = df_ar['duration'].apply(lambda x: x.total_seconds()/86400)
    df_ar['event'] = df_ar['event'].astype(int)

    df_ar = df_ar.drop(columns=['ddeb_complic', 'Date of transplantation'])
    return df_ar

def plot_kaplan_meier(df_ar, kmf, group, dict_targets, dict_target_thresholds, years, save=False, filename=None):
    T = df_ar["duration"]
    E = df_ar["event"]
    targets = dict_targets[group]
    target_thresholds = dict_target_thresholds[group]
    fig, ax = plt.subplots(len(targets), 3, figsize=(3*5,5*len(targets))) 
    for i, target in enumerate(targets):
        for j, target_threshold in enumerate(target_thresholds[target]):
            #print((target, target_threshold))
            mask = df_ar[target] > target_threshold
            kmf.fit(T[mask], event_observed=E[mask], label="{} target > {:.2f}".format(target, target_threshold))
            kmf.plot_survival_function(ax=ax[i,j])

            mask = df_ar[target] <= target_threshold
            kmf.fit(T[mask], event_observed=E[mask], label="{} target <= {:.2f}".format(target, target_threshold))
            kmf.plot_survival_function(ax=ax[i,j])

            ax[i,j].set_ylabel("est. probability of survival")
            ax[i,j].set_xlabel("time $t$ (days)")
            ax[i,j].legend(loc="best")
            ax[i,j].set_xlim((0,years*365))
            ax[i,j].set_ylim((0.,1.05))

            results = logrank_test(T[mask], T[~mask], E[mask], E[~mask], alpha=.99)
            ax[i,j].set_title('test_statistic: {:.2f} | p: {:.4f} | -log2(p): {:.4f}'.format(results.summary['test_statistic'].values[0], 
                                                                               results.summary['p'].values[0], 
                                                                               results.summary['-log2(p)'].values[0]))
    plt.tight_layout()
    if save:
            fig.savefig(os.path.join(save, 'kaplan_meier_ar_{}'.format(group)+filename))

def fit_and_score_features(df_cr, features, train_patients, test_patients, n_splits=3):
    n_features = len(features)
    np_scores = np.empty((n_features, 3))
    cph = CoxPHFitter()
    skf = StratifiedKFold(n_splits=n_splits)
    for j, feat in enumerate(features):
        print(feat)
        Xj_train = df_cr[df_cr['patient'].isin(train_patients)]
        Xj_test = df_cr[df_cr['patient'].isin(test_patients)]
        Xj = Xj_train[[feat, 'event', 'duration']].dropna(subset=[feat])
        Xj_test = Xj_test[[feat, 'event', 'duration']].dropna(subset=[feat])
        train_scores, val_scores, test_scores = np.empty(n_splits), np.empty(n_splits), np.empty(n_splits)
        for i, (train_index, val_index) in enumerate(skf.split(Xj, Xj[['event']])):
            X_train, X_val = Xj.iloc[train_index], Xj.iloc[val_index]
            try:
                cph.fit(X_train, duration_col='duration', event_col='event')
                train_scores[i] = cph.concordance_index_
                val_scores[i] = cph.score(X_val, scoring_method='concordance_index')
                test_scores[i] = cph.score(Xj_test, scoring_method='concordance_index')
            except Exception as ex:
                train_scores[i] = 0
                val_scores[i] = 0  
                test_scores[i] = 0
        np_scores[j,0] = train_scores.mean()
        np_scores[j,1] = val_scores.mean()
        np_scores[j,2] = test_scores.mean()
    return np_scores

def cross_validate_coxph(X, penalizers, l1_ratios, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits)
    print('penalizers', penalizers)
    print('l1_ratios', l1_ratios)

    scores = {}
    for penalizer in penalizers:
        for l1_ratio in l1_ratios:
            print('penalizer_{:.0e}_l1_ratio_{:.1f}'.format(penalizer, l1_ratio))
            cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
            train_scores, val_scores = np.empty(n_splits), np.empty(n_splits)
            for i, (train_index, val_index) in enumerate(skf.split(X, X[['event']])):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                try:    
                    cph.fit(X_train, duration_col='duration', event_col='event')
                    train_scores[i] = cph.concordance_index_
                    val_scores[i] = cph.score(X_val, scoring_method='concordance_index')
                except Exception as ex:
                    train_scores[i] = 0
                    val_scores[i] = 0 
            scores['penalizer_{:.0e}_l1_ratio_{:.1f}'.format(penalizer, l1_ratio)] = [train_scores.mean(), 
                                                                              train_scores.std(),
                                                                              val_scores.mean(), 
                                                                              val_scores.std()]

    scores_df = pd.DataFrame.from_dict(scores, orient='index', columns=['train_mean', 'train_std', 'val_mean', 'val_std'])
    return scores_df

def re_cross_validate_coxph(X, df_cph, penalizers, l1_ratios, thresholds, thresholds_ids, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits)
    print('thresholds', thresholds)
    print('penalizers', penalizers)
    print('l1_ratios', l1_ratios)
    
    scores = {}
    for threshold, thresholds_id in zip(thresholds, thresholds_ids):
        idx = df_cph[df_cph['coef'].abs()>threshold].index
        idx = list(idx)
        Xt = X[list(idx)+['event', 'duration']]
        for penalizer in penalizers:
            for l1_ratio in l1_ratios:
                print('threshold_{}_penalizer_{:.0e}_l1_ratio_{:.1f}'.format(thresholds_id, penalizer, l1_ratio))
                cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
                train_scores, val_scores = np.empty(n_splits), np.empty(n_splits)
                for i, (train_index, val_index) in enumerate(skf.split(Xt, Xt[['event']])):
                    X_train, X_val = Xt.iloc[train_index], Xt.iloc[val_index]
                    try:    
                        cph.fit(X_train, duration_col='duration', event_col='event')
                        train_scores[i] = cph.concordance_index_
                        val_scores[i] = cph.score(X_val, scoring_method='concordance_index')
                    except Exception as ex:
                        train_scores[i] = 0
                        val_scores[i] = 0 
                scores['threshold_{}_penalizer_{:.0e}_l1_ratio_{:.1f}'.format(thresholds_id, penalizer, l1_ratio)] = [train_scores.mean(), 
                                                                                  train_scores.std(),
                                                                                  val_scores.mean(), 
                                                                                  val_scores.std()]

    scores_df = pd.DataFrame.from_dict(scores, orient='index', columns=['train_mean', 'train_std', 'val_mean', 'val_std'])
    return scores_df

def plot_cv_results(cv_results, param_x, param_z):
    """
    cv_results - cv_results_ attribute of a GridSearchCV instance (or similar)
    param_x - name of grid search parameter to plot on x axis
    param_z - name of grid search parameter to plot by line color
    """
    col_x = 'param_' + param_x
    col_z = 'param_' + param_z
    df_plot = cv_results.groupby([col_x, col_z]).mean()
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))
    lines = []
    for col_z_value in df_plot.index.unique(col_z):
        x = df_plot.xs(col_z_value, level=col_z).index.to_numpy()
        y = df_plot.xs(col_z_value, level=col_z)['mean_test_score'].to_numpy()
        y_stds = df_plot.xs(col_z_value, level=col_z)['std_test_score'].to_numpy()
        line, = ax.plot(x, y, label=col_z_value)
        lines.append(line)
        ax.fill_between(x, y - y_stds, y + y_stds, alpha=.15)
    ax.set_title("CV Grid Search Results")
    ax.set_xscale("log")
    ax.set_xlabel(param_x)
    ax.set_ylabel("concordance index")
    ax.legend(handles=lines, title=param_z)
    plt.gcf().patch.set_facecolor('white')
    return fig

def plot_cv_results2(cv_results, param_x):
    """
    cv_results - cv_results_ attribute of a GridSearchCV instance (or similar)
    param_x - name of grid search parameter to plot on x axis
    param_z - name of grid search parameter to plot by line color
    """
    col_x = 'param_' + param_x
    df_plot = cv_results.groupby([col_x]).mean()
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))
    lines = []
    x = df_plot.index.to_numpy()
    y = df_plot['mean_test_score'].to_numpy()
    y_stds = df_plot['std_test_score'].to_numpy()
    line, = ax.plot(x, y)
    lines.append(line)
    ax.fill_between(x, y - y_stds, y + y_stds, alpha=.15)
    ax.set_title("CV Grid Search Results")
    ax.set_xscale("log")
    ax.set_xlabel(param_x)
    ax.set_ylabel("concordance index")
    plt.gcf().patch.set_facecolor('white')
    return fig


