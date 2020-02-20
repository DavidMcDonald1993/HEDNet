#!/bin/bash

#SBATCH --job-name=ATPRECON
#SBATCH --output=ATPRECON_%A_%a.out
#SBATCH --error=ATPRECON_%A_%a.err
#SBATCH --array=0-2249
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=25G

datasets=(cora_ml citeseer pubmed email wiki_vote)
dims=(2 5 10 25 50)
seeds=({00..29})
methods=(linear ln harmonic)
exp=recon_experiment

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

data_dir=datasets/${dataset}
edgelist=${data_dir}/edgelist.tsv.gz
embedding_dir=$(printf "../atp/embeddings/${dataset}/${exp}/seed=%03d/dim=%03d/${method}" ${seed} ${dim})

test_results=$(printf \
    "test_results/${dataset}/${exp}/dim=%03d/${method}/" ${dim})
echo ${embedding_dir}
echo ${test_results}

args=$(echo --edgelist ${edgelist} --dist_fn st \
    --embedding ${embedding_dir} --seed ${seed} \
    --test-results-dir ${test_results})
echo ${args}

module purge
module load bluebear
module load apps/python3/3.5.2

python evaluate_reconstruction.py ${args}
