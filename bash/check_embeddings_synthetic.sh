#!/bin/bash

# experiments
for dataset in {00..29}
do
	for dim in 2 5 10 25 50
	do	
		for seed in 0
		do
			for exp in recon_experiment lp_experiment
			do
				embedding_dir=$(printf \
				"embeddings/synthetic_scale_free/${dataset}/${exp}/seed=%03d/dim=%03d/" ${seed} ${dim})

				if [ ! -f ${embedding_dir}final_embedding.csv ] 
				then
					echo no embedding at ${embedding}${embedding_dir}final_embedding.csv
				fi

				if [ ! -f ${embedding_dir}final_variance.csv ]
				then 
					echo no variance at ${embedding_dir}final_variance.csv
				fi
			done
		done
	done
done