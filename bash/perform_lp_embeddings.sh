#!/bin/bash

#SBATCH --job-name=embeddingsLP
#SBATCH --output=embeddingsLP_%A_%a.out
#SBATCH --error=embeddingsLP_%A_%a.err
#SBATCH --array=0-749
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=3
#SBATCH --mem=20G

e=10000

datasets=({cora_ml,citeseer,pubmed,wiki_vote,email})
dims=(2 5 10 25 50)
seeds=({0..29})
# seeds=(0)

num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}

dataset_id=$((SLURM_ARRAY_TASK_ID / (num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / num_seeds % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID % num_seeds))

dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}

data_dir=datasets/${dataset}
edgelist=$(printf edgelists/${dataset}/seed=%03d/training_edges/edgelist.tsv ${seed})
embedding_dir=embeddings/${dataset}/lp_experiment
embedding_dir=$(printf "${embedding_dir}/seed=%03d/dim=%03d/" ${seed} ${dim})

if [ ! -f ${embedding_dir}final_embedding.csv ]
then 
    module purge
    module load bluebear
    module load TensorFlow/1.10.1-foss-2018b-Python-3.6.6
    pip install --user keras==2.2.4

    args=$(echo --edgelist ${edgelist} \
    --embedding ${embedding_dir} --no-walks --seed ${seed} \
    --dim ${dim} --use-generator --context-size 1 -e ${e})

    python main.py ${args}
fi