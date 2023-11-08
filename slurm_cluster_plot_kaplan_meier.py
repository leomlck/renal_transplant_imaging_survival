import os
import io
import pandas as pd
import time

job_name = '/gpfs/users/mileckil/kidney_workspace/project_kidney/survival_workspace/src/fit_job.sh'

start_script = ('#!/bin/bash\n' +
                '#SBATCH --job-name=main_kp\n' +
                '#SBATCH --output=output/%x.o%j\n' +
                '#SBATCH --time=02:00:00\n' +
                '#SBATCH --ntasks=1\n' +
                '#SBATCH --cpus-per-task=4\n'
                '#SBATCH --mem=2GB\n' +
                '#SBATCH --partition=cpu_med\n' +
                '#SBATCH --export=NONE\n' +
                '\n' +
                'module load anaconda3/2021.05/gcc-9.2.0\n'+
                'source activate survenv\n')



command = 'python launch_cluster_plot_kaplan_meier.py'

with open(job_name, 'w') as fh:
    fh.write(start_script)
    fh.write(command)
stdout = pd.read_csv(io.StringIO(os.popen("sbatch " + job_name).read()), delim_whitespace=True)
print(stdout)
os.remove(job_name)

