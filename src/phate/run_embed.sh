#!/bin/bash

SBATCH_OPTS="\
--mem=64GB \
--partition=preempt \
--mail-type=END --mail-user=Kapil.Devkota@tufts.edu \
--time=1-10:00:00 \
"

DIM=2
VERBOSE=
TIMESTEPS=5
while getopts "n:d:o:t:v" args; do
    case $args in
	n) NETWORK=$OPTARG
	   ;;
	d) DIM=$OPTARG
	   ;;
	o) OUTPUT=$OPTARG
	   ;;
	t) TIMESTEPS=$OPTARG
	   ;;
	v) VERBOSE=--verbose
	   ;;
    esac
done

OP=${OUTPUT}-LOG.txt
sbatch $SBATCH_OPTS -o ${OP} ./phate_main.py --network_file=$NETWORK --output_prefix=$OUTPUT --n_dims=$DIM --timesteps=$TIMESTEPS $VERBOSE
