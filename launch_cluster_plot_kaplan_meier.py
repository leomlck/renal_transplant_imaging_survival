import os
import io
import pandas as pd
import time
import subprocess
from sklearn.model_selection import ParameterGrid

selected = 1
ext = 1

parameters = {
'model': ['cnn'],
'mod': ['dce'],
'pretraining': ['patient5k'], 
'event': ['cr'],
}

if selected:
    parameters['survival_model'] = ['coxnet']
if not mix:
    if not ext or selected:
        parameters['exam'] = ['J15']


base_command = 'python cluster_plot_kaplan_meier'
if ext:
    base_command += '_ext'
if selected:
    base_command += '_selected'
base_command += '.py '

for i, params in enumerate(list(ParameterGrid(parameters))):
    print('Sending job params {}/{}'.format(i+1, len(list(ParameterGrid(parameters)))))
    params_list = ['--{} {}'.format(param, params[param]) for param in params.keys()]
    command = base_command + ' '.join(params_list)
    subprocess.Popen(command, shell=True).wait()

