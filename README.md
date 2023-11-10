# Survival Analysis from renal transplants imaging features

Assuming you have your best imaging features you want to perform survival analysis with (for example obtained with github.com/leomlck/renal_transplant_imaging).

## Usage 

Cross-validate your Cox model on your main data features (identified by *model*, *mod*, *exam* and *pretraining* arguments) and test on the '_external' features with:
```
python fit_coxnet.py --model cnn --mod dce --pretraining patient --exam D15 --event cr
```

Use parameters grid and slurm jobs to validate different set of features (modify the parameters directly in the ```slurm_fit_coxnet.py``` file):
```
python slurm_fit_coxnet.py
```
Or make a config file from ```config/make_args_settings_file.py``` to directly send slurm job arrays from:
```
python slurm_jarrays_fit_coxnet.py
```

Cluster your features in two groups and plot the corresponding Kaplan-Meier curves regarding your survival event, use instead ```cluster_plot_kaplan_meier_selected.py``` to only cluster from the selected features with your trained Cox model:
```
python cluster_plot_kaplan_meier.py --model cnn --mod dce --pretraining patient --exam D15 --event cr
```

Use slurm jobs to send multiple scripts (modify the parameters directly in the ```launch_cluster_plot_kaplan_meier.py``` file) with:
```
python slurm_cluster_plot_kaplan_meier.py
```
