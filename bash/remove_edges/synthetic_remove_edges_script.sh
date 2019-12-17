#!/bin/bash

#SBATCH --job-name=removeEdgesSynthetic
#SBATCH --output=removeEdgesSynthetic_%A_%a.out
#SBATCH --error=removeEdgesSynthetic_%A_%a.err
#SBATCH --array=0-2999
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20G

datasets=({00..99})
seeds=({0..29})

num_datasets=${#datasets[@]}
num_seeds=${#seeds[@]}

dataset_id=$((SLURM_ARRAY_TASK_ID / num_seeds % num_datasets ))
seed_id=$((SLURM_ARRAY_TASK_ID % (num_seeds) ))

dataset=${datasets[$dataset_id]}
seed=${seeds[$seed_id]}

edgelist=datasets/synthetic_scale_free/${dataset}/edgelist.tsv
output=edgelists/synthetic_scale_free/${dataset}

edgelist_f=$(printf "${output}/seed=%03d/training_edges/edgelist.tsv" ${seed} )

if [ ! -f $edgelist_f  ]
then
	module purge
	module load bluebear
	module load apps/python3/3.5.2

	python remove_edges.py --edgelist=$edgelist --output=$output --seed $seed
fi