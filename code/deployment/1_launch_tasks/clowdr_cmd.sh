#!/bin/bash

datapath=/project/6008063/gkiar/data/RocklandSample/derivatives/
baseopts="-v ${datapath}:${datapath} --cluster slurm --clusterargs account:rpp-aevans-ab,time:23:00:00,mem:8192 -V --simg /home/gkiar/images/dipy_tracking_v0.4.0-patch1.sif "

#--------------------------------------------

# Variance: Multi Pipeline
### Generate invocations
python create_invocations.py ${datapath}/preproc_ds ${datapath}/connectomes_mp example.json 20 multi_pipeline

### Launch reference tasks
invodir="./invocations-multi_pipeline-ref"
clowdir="/home/gkiar/executions/nkirs/multi_pipeline/ref/"
clowdr local dipy_tracking.json ${invodir} ${clowdir} ${baseopts} -g 20 --sweep "prob"

### Launch simulation tasks
invodir="./invocations-multi_pipeline"
clowdir="/home/gkiar/executions/nkirs/multi_pipeline/mca/"
clowdr local dipy_tracking.json ${invodir} ${clowdir} ${baseopts} -g 5 --sweep "prob" --sweep "output_directory"

#--------------------------------------------

# Variance: Multi Seed
### Generate invocations
python create_invocations.py ${datapath}/preproc_ds ${datapath}/connectomes_ms example.json 10 multi_seed

### Launch reference tasks
invodir="./invocations-multi_seed-ref"
clowdir="/home/gkiar/executions/nkirs/multi_seed/ref/"
echo clowdr local dipy_tracking.json ${invodir} ${clowdir} ${baseopts} -g 20 --sweep "random_seed"

### Launch simulation tasks
invodir="./invocations-multi_seed"
clowdir="/home/gkiar/executions/nkirs/multi_seed/mca/"
echo clowdr local dipy_tracking.json ${invodir} ${clowdir} ${baseopts} -g 5 --sweep "random_seed" --sweep "output_directory"

#--------------------------------------------

# Signal: Age/BMI
### Generate invocations
python create_invocations.py ${datapath}/preproc_100 ${datapath}/connectomes example.json 20 age_bmi

### Launch reference tasks
invodir="./invocations-age_bmi-ref"
clowdir="/home/gkiar/executions/nkirs/age_bmi/ref/"
clowdr local dipy_tracking.json ${invodir} ${clowdir} ${baseopts} -g 20 --sweep "prob"

### Launch simulation tasks
invodir="./invocations-age_bmi"
clowdir="/home/gkiar/executions/nkirs/age_bmi/mca/"
clowdr local dipy_tracking.json ${invodir} ${clowdir} ${baseopts} -g 5 --sweep "prob" --sweep "output_directory"

#--------------------------------------------

