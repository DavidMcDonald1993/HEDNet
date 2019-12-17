#!/bin/bash

#SBATCH --job-name=HEDNETevaluateLP
#SBATCH --output=HEDNETevaluateLP_%A_%a.out
#SBATCH --error=HEDNETevaluateLP_%A_%a.err
#SBATCH --array=0-749
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=30G

datasets=({cora_ml,citeseer,pubmed,wiki_vote,email})
dims=(2 5 10 25 50)
seeds=({0..29})
exp=lp_experiment

num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}

dataset_id=$((SLURM_ARRAY_TASK_ID / (num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / num_seeds % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID % num_seeds ))

dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}

data_dir=datasets/${dataset}
edgelist=${data_dir}/edgelist.tsv
embedding_dir=embeddings/${dataset}/${exp}
output=edgelists/${dataset}

test_results=$(printf \
    "test_results/${dataset}/${exp}/dim=%03d/HEDNet/" ${dim})
embedding_dir=$(printf \
    "${embedding_dir}/seed=%03d/dim=%03d" ${seed} ${dim})
echo ${embedding_dir}

args=$(echo --edgelist ${edgelist} --output ${output} --dist_fn klh \
    --embedding ${embedding_dir} --seed ${dataset} \
    --test-results-dir ${test_results})
echo ${args}

module purge
module load bluebear
module load apps/python3/3.5.2

python evaluation/evaluate_lp.py ${args}
