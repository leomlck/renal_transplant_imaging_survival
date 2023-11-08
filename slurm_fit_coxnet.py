import os
import io
import pandas as pd
import time
from sklearn.model_selection import ParameterGrid

job_name = '/gpfs/users/mileckil/kidney_workspace/renal_transplant_imaging_survival/fit_job_coxnet.sh'

start_script = ('#!/bin/bash\n' +
                '#SBATCH --job-name=main_survival\n' +
                '#SBATCH --output=output/%x.o%j\n' +
                '#SBATCH --time=02:00:00\n' +
                '#SBATCH --ntasks=1\n' +
                '#SBATCH --cpus-per-task=32\n'
                '#SBATCH --mem=6GB\n' +
                '#SBATCH --partition=cpu_med\n' +
                '#SBATCH --export=NONE\n' +
                '\n' +
                'module load anaconda3/2021.05/gcc-9.2.0\n'+
                'source activate survenv\n')

parameters = {
'model': ['cnn'],
'mod': ['dce'],
'pretraining': ['patient5k'],
'exam': ['J15'],
'event': ['cr'],
'seed': [42],
}

for i, params in enumerate(list(ParameterGrid(parameters))):
    params_list = ['--{} {}'.format(param, params[param]) for param in params.keys()]
    command = 'python fit_coxnet.py ' + ' '.join(params_list) 
    with open(job_name, 'w') as fh:
        fh.write(start_script)
        fh.write(command)
    stdout = pd.read_csv(io.StringIO(os.popen("sbatch " + job_name).read()), delim_whitespace=True)
    print(stdout)
    os.remove(job_name)
    time.sleep(20)
