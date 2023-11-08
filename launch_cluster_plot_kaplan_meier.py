import os
import io
import pandas as pd
import time
import subprocess
from sklearn.model_selection import ParameterGrid

selected = 1
ext = 1
mix = 0

parameters = {
'model': ['cnn'],
'mod': ['dce'],
'pretraining': ['patient5k', 'fibrose5k', 'incomp5k', 'ischem5k', 'mdrd5k', 'agedonor5k'], #['patient_all', 'agedonor_all', 'mdrd_all', 'fibrose_all', 'incomp_all', 'ischem_all'], #['patient5k', 'fibrose5k', 'incomp5k', 'ischem5k', 'mdrd5k', 'agedonor5k'], #, 'mdrd5k_mresnet18', 'mdrd5k_r18vits16'],['patient5k_ext_nocur', 'patient2k5_ext'], #
'event': ['cr'], #, 'cr'], #['ar', 'cr'],
}

if selected:
    parameters['survival_model'] = ['coxnet']
if not mix:
    if not ext or selected:
        parameters['exam'] = ['J15']#, 'J30', 'M3', 'M12']


base_command = 'python cluster_plot_kaplan_meier'
if ext:
    base_command += '_ext'
if selected:
    base_command += '_selected'
if not mix:
    base_command += '.py '
else:
    base_command += '2.py '

for i, params in enumerate(list(ParameterGrid(parameters))):
    print('Sending job params {}/{}'.format(i+1, len(list(ParameterGrid(parameters)))))
    params_list = ['--{} {}'.format(param, params[param]) for param in params.keys()]
    command = base_command + ' '.join(params_list)
    subprocess.Popen(command, shell=True).wait()

