#!/bin/bash

SBATCH_OPTS="\
--mem=128GB \
--partition=preempt \
--mail-type=END --mail-user=Kapil.Devkota@tufts.edu \
--time=2-00:00:00 \
"
FOLDER=
DIM=1000
CONST=2
while getopts "f:d:c:" args; do
    case $args in
	f) FOLDER=${OPTARG}
	   ;;
	d) DIM=${OPTARG}
	   ;;
	c) CONST=${OPTARG}
	   ;;
    esac
done

if [ ! -d sbatch_logs ]
then
    mkdir sbatch_logs
fi

log=sbatch_logs/DREAM_OUT_DIM_${DIM}_C_${CONST}.log

sbatch $SBATCH_OPTS -o $log ./updated_mashup.py --input_folder=${FOLDER} --dims=${DIM} --const=${CONST} --verbose 
