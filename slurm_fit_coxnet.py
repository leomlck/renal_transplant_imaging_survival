import os
import io
import pandas as pd
import time
from sklearn.model_selection import ParameterGrid

job_name = '/gpfs/users/mileckil/kidney_workspace/project_kidney/survival_workspace/src/fit_job_coxnet.sh'

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
'mod': ['dce', 'dw'],
'pretraining': ['patient5k', 'incomp5k', 'ischem5k', 'agedonor5k', 'fibrose5k', 'mdrd5k'], #'patient5k_ext_nocur', 'patient2k5_ext'], #['radiomics_binWidth_25'], #['patient5k_ext_nocur', 'patient2k5_ext'], #'patient5k', 'incomp5k', 'ischem5k', 'agedonor5k', 'fibrose5k', 'mdrd5k_mresnet18', 'mdrd5k_r18vits16', 'patient_all', 'agedonor_all', 'mdrd_all', 'fibrose_all', 'incomp_all', 'ischem_all'], #['radiomics_binWidth_25',
'exam': ['J15', 'J30', 'M3', 'M12'],
'event': ['cr'], #'cr'],
'seed': [42],
}
#time.sleep(60*60*3)
for i, params in enumerate(list(ParameterGrid(parameters))):
    params_list = ['--{} {}'.format(param, params[param]) for param in params.keys()]
    command = 'python fit_coxnet_30.py ' + ' '.join(params_list) 
    with open(job_name, 'w') as fh:
        fh.write(start_script)
        fh.write(command)
    stdout = pd.read_csv(io.StringIO(os.popen("sbatch " + job_name).read()), delim_whitespace=True)
    print(stdout)
    os.remove(job_name)
    time.sleep(20)
    '''
    if (i+1)%25==0:
        time.sleep(60*25)
    '''
