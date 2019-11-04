#!/bin/bash

datasets=({00..29})
seeds=({0..29})

for dataset in ${datasets[@]};
do

	output=edgelists/synthetic_scale_free/${dataset}

	for seed in ${seeds[@]};
	do
		edgelist_f=$(printf "${output}/seed=%03d/training_edges/edgelist.tsv" ${seed} )

		if [ ! -f $edgelist_f  ]
		then
			echo $edgelist_f is missing 
		fi

	done
done

# num_datasets=${#datasets[@]}
# num_seeds=${#seeds[@]}

# dataset_id=$((SLURM_ARRAY_TASK_ID / num_seeds % num_datasets ))
# seed_id=$((SLURM_ARRAY_TASK_ID % (num_seeds) ))

# dataset=${datasets[$dataset_id]}
# seed=${seeds[$seed_id]}

# edgelist=datasets/synthetic_scale_free/${dataset}/edgelist.tsv
# output=edgelists/synthetic_scale_free/${dataset}

# edgelist_f=$(printf "${output}/seed=%03d/training_edges/edgelist.tsv" ${seed} )

# if [ ! -f $edgelist_f  ]
# then
# 	module purge
# 	module load bluebear
# 	module load apps/python3/3.5.2

# 	python remove_edges.py --edgelist=$edgelist --output=$output --seed $seed
# fi