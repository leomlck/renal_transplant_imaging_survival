import os
import io
import pandas as pd
import time
import subprocess
from sklearn.model_selection import ParameterGrid

# Use only the features selected by the survival model (or not)
selected = 1

# Features origin parameters
parameters = {
'model': ['cnn'],
'mod': ['dce'],
'pretraining': ['patient5k'], 
'event': ['cr'],
}

if selected:
    parameters['survival_model'] = ['coxnet']


base_command = 'python cluster_plot_kaplan_meier'
if selected:
    base_command += '_selected'
base_command += '.py '

for i, params in enumerate(list(ParameterGrid(parameters))):
    print('Sending job params {}/{}'.format(i+1, len(list(ParameterGrid(parameters)))))
    params_list = ['--{} {}'.format(param, params[param]) for param in params.keys()]
    command = base_command + ' '.join(params_list)
    subprocess.Popen(command, shell=True).wait()

