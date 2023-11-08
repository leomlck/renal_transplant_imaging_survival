import pandas as pd
from sklearn.model_selection import ParameterGrid

parameters = {
'model': ['radiomics'],
'mod': ['dce'], #'dw'],
'pretraining': ['radiomics_binWidth_25'], #['patient5k', 'incomp5k', 'ischem5k', 'agedonor5k', 'fibrose5k', 'mdrd5k'], #'patient5k_ext_nocur', 'patient2k5_ext'], #['radiomics_binWidth_25'], #['patient5k_ext_nocur', 'patient2k5_ext'], #'patient5k', 'incomp5k', 'ischem5k', 'agedonor5k', 'fibrose5k', 'mdrd5k_mresnet18', 'mdrd5k_r18vits16', 'patient_all', 'agedonor_all', 'mdrd_all', 'fibrose_all', 'incomp_all', 'ischem_all'], #['radiomics_binWidth_25',
'exam': ['J15'], #, 'J30', 'M3', 'M12'],
'event': ['cr'], #'cr'],
'clinical': [0, 1],
'seed': [8],
}

params = list(ParameterGrid(parameters))
df_params = pd.DataFrame(params)
print('Number of settings :', len(df_params))

df_params.to_csv('df_args_settings_radiomics_J15_all_{}.csv'.format(len(df_params)), index=False)
#df_params.to_csv('df_args_settings_net.csv', index=False)
#df_params.to_csv('df_args_settings_netph.csv', index=False)

