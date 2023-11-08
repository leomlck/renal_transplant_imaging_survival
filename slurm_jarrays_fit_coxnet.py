import os
import io
import pandas as pd
import time
from sklearn.model_selection import ParameterGrid

settings = 'radiomics_J15_all_2'
#settings = 'dce_J15_6'
n_arrays = settings.split('_')[-1]

job_name = '/gpfs/users/mileckil/kidney_workspace/project_kidney/survival_workspace/src/fit_job_coxnet.sh'

start_script = ('#!/bin/bash\n' +
                '#SBATCH --job-name=main_survival_net\n' +
                '#SBATCH --output=output/%x.o%A_%a\n' +
                '#SBATCH --time=03:00:00\n' +
                '#SBATCH --ntasks=1\n' +
                '#SBATCH --cpus-per-task=32\n' +
                '#SBATCH --array=1-{}\n'.format(n_arrays) +
                '#SBATCH --mem=6GB\n' +
                '#SBATCH --partition=cpu_med\n' +
                '#SBATCH --export=NONE\n' +
                '\n' +
                'module load anaconda3/2021.05/gcc-9.2.0\n'+
                'source activate survenv\n')

command = 'python fit_coxnet_30_jarrays.py --settings {} --job_index $SLURM_ARRAY_TASK_ID'.format(settings)  
with open(job_name, 'w') as fh:
    fh.write(start_script)
    fh.write(command)
stdout = pd.read_csv(io.StringIO(os.popen("sbatch " + job_name).read()), delim_whitespace=True)
print(stdout)
os.remove(job_name)

