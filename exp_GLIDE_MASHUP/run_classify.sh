#!/bin/bash

SBATCH_OPTS="\
--mem=128GB \
--partition=largemem \
--mail-type=END --mail-user=Kapil.Devkota@tufts.edu \
--time=2-00:00:00 \
"
ODIR=slurm_results
MINP=50
MINL=5
MAXL=5

GOTYPE=(P F C)

MODEL_DIR=dream_dsd_folder
EMBED=OUT-buggy_dim_1000.npy

while getopts "e:" args; do
    case $args in
	e) EMBED=$OPTARG
	   ;;
    esac
done

OFILE=output_folder/$EMBED
if [ ! -d $OFILE ]; then mkdir $OFILE; echo "Output folder created"; fi



for GOT in ${GOTYPE[@]}
do
    OUTPUT=$OFILE/GO_${GOT}.txt
    sbatch $SBATCH_OPTS -o ${OUTPUT} ./classify.py -v  --go_type=${GOT} --network=${MODEL_DIR}/${EMBED} --json=${MODEL_DIR}/nodemap.json 
done 
