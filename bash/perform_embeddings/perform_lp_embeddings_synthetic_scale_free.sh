#!/bin/bash

#SBATCH --job-name=HEDNETembeddingsLPSynthetic
#SBATCH --output=HEDNETembeddingsLPSynthetic_%A_%a.out
#SBATCH --error=HEDNETembeddingsLPSynthetic_%A_%a.err
#SBATCH --array=0-149
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=3
#SBATCH --mem=16G

e=1000

datasets=({00..29})
dims=(2 5 10 25 50)
seeds=(0)

num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}

dataset_id=$((SLURM_ARRAY_TASK_ID / (num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / num_seeds % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID % num_seeds))

dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}

data_dir=datasets/synthetic_scale_free/${dataset}
edgelist=$(printf edgelists/synthetic_scale_free/${dataset}/seed=%03d/training_edges/edgelist.tsv ${seed})
embedding_dir=embeddings/synthetic_scale_free/${dataset}/lp_experiment
embedding_dir=$(printf "${embedding_dir}/seed=%03d/dim=%03d/" ${seed} ${dim})

if [ ! -f ${embedding_dir}final_embedding.csv.gz ]
then 
    module purge
    module load bluebear
    module load TensorFlow/1.10.1-foss-2018b-Python-3.6.6
    pip install --user keras==2.2.4

    args=$(echo --edgelist ${edgelist} \
    --embedding ${embedding_dir}  --seed ${seed} \
    --dim ${dim} --context-size 1 -e ${e})

    python main.py ${args}
fi