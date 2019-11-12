#!/bin/bash

#SBATCH --job-name=embeddingsRECONSynthetic
#SBATCH --output=embeddingsRECONSynthetic_%A_%a.out
#SBATCH --error=embeddingsRECONSynthetic_%A_%a.err
#SBATCH --array=0-3599
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=16G

e=1000

datasets=({00..29})
dims=(5 10 25 50)
seeds=({0..29})

num_datasets=${#datasets[@]}
num_dims=${#dims[@]}
num_seeds=${#seeds[@]}

dataset_id=$((SLURM_ARRAY_TASK_ID / (num_seeds * num_dims) % num_datasets))
dim_id=$((SLURM_ARRAY_TASK_ID / num_seeds % num_dims))
seed_id=$((SLURM_ARRAY_TASK_ID % num_seeds ))

dataset=${datasets[$dataset_id]}
dim=${dims[$dim_id]}
seed=${seeds[$seed_id]}

data_dir=datasets/synthetic_scale_free/${dataset}
edgelist=${data_dir}/edgelist.tsv
# features=${data_dir}/feats.csv
# labels=${data_dir}/labels.csv
embedding_dir=embeddings/synthetic_scale_free/${dataset}/recon_experiment
# walks_dir=walks/${dataset}/recon_experiment

embedding_dir=$(printf "${embedding_dir}/seed=%03d/dim=%03d/" ${seed} ${dim})
# echo $embedding_dir

# if [ ! -f embedding_f ]
# then 
module purge
module load bluebear
# module load apps/python3/3.5.2
# module load apps/keras/2.0.8-python-3.5.2
module load TensorFlow/1.10.1-foss-2018b-Python-3.6.6
pip install --user keras==2.2.4

args=$(echo --edgelist ${edgelist} \
--embedding ${embedding_dir} --no-walks --seed ${seed} \
--dim ${dim} --use-generator --context-size 1 -e ${e})

# echo $args

python main.py ${args}
# fi