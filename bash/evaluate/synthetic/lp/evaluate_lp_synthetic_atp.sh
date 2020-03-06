#!/bin/bash

#SBATCH --job-name=ATPSYNLP
#SBATCH --output=ATPSYNLP_%A_%a.out
#SBATCH --error=ATPSYNLP_%A_%a.err
#SBATCH --array=0-299
#SBATCH --time=30:00
#SBATCH --ntasks=1
#SBATCH --mem=5G

datasets=({0..29})
dims=(2 5 10 25 50)
seeds=(0)
methods=(ln harmonic)
exp=lp_experiment

num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}
num_methods=${#methods[@]}

dataset_id=$((SLURM_ARRAY_TASK_ID / (num_methods * num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / (num_methods * num_seeds) % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID / num_methods % num_seeds ))
method_id=$((SLURM_ARRAY_TASK_ID % num_methods ))

dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}
method=${methods[$method_id]}

echo $dataset $dim $seed $method

data_dir=$(printf datasets/synthetic_scale_free/%02d ${dataset})
edgelist=${data_dir}/edgelist.tsv.gz
embedding_dir=$(printf "../atp/embeddings/synthetic_scale_free/%02d/${exp}/seed=%03d/dim=%03d/${method}" ${dataset} ${seed} ${dim})

removed_edges_dir=$(printf edgelists/synthetic_scale_free/%02d/seed=%03d/removed_edges ${dataset} ${seed})

test_results=$(printf \
    "test_results/synthetic_scale_free/${exp}/dim=%03d/${method}/" ${dim})
echo ${embedding_dir}
echo ${test_results}

args=$(echo --edgelist ${edgelist} \
    --removed_edges_dir ${removed_edges_dir} \
    --dist_fn st \
    --embedding ${embedding_dir} --seed ${dataset} \
    --test-results-dir ${test_results})
echo ${args}

module purge
module load bluebear
module load apps/python3/3.5.2

python evaluate_lp.py ${args}
