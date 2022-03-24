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

epoch=50
MODEL_DIR=dream_nets
CFILE=("OUT-dim_1000_const_2.0" "OUT-dim_500_const_2.0" "OUT-dim_500_const_3.0.npy") # DREAM1,2,3
for GOT in ${GOTYPE[@]}
do
    for CF in  ${CFILE[@]}
    do
	echo "HERE"
	OUTPUT=dream_${CF}_GO_${GOT}.txt
	sbatch $SBATCH_OPTS -o ${OUTPUT} ./classify.py -v  --go_type=${GOT} --network=${MODEL_DIR}/${CF}.npy --json=${MODEL_DIR}/dream_nodemap.json 
    done
done 
