#!/bin/bash

#SBATCH --job-name=LINESYNRECON
#SBATCH --output=LINESYNRECON_%A_%a.out
#SBATCH --error=LINESYNRECON_%A_%a.err
#SBATCH --array=0-119
#SBATCH --time=20:00
#SBATCH --ntasks=1
#SBATCH --mem=5G

datasets=({0..29})
dims=(5 10 25 50)
seeds=(0)
methods=(line)
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

echo $dataset $dim $seed $method

data_dir=$(printf datasets/synthetic_scale_free/%02d ${dataset})
edgelist=${data_dir}/edgelist.tsv.gz
embedding_dir=$(printf ../OpenANE/embeddings/synthetic_scale_free/%02d/${exp}/$((2*dim))/${method}/${seed} ${dataset})

test_results=$(printf \
    "test_results/synthetic_scale_free/${exp}/dim=%03d/${method}/" ${dim})
echo ${embedding_dir}
echo ${test_results}

args=$(echo --edgelist ${edgelist} --dist_fn euclidean \
    --embedding ${embedding_dir} --seed ${dataset} \
    --test-results-dir ${test_results})
echo ${args}

module purge
module load bluebear
module load apps/python3/3.5.2

python evaluate_reconstruction.py ${args}
