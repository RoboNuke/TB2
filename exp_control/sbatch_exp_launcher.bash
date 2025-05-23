#!/bin/bash

num_agents=4
num_exp_per_env=2
num_exp_env=1
names=("Obs" "DMP_Obs" "Hist_Obs" "Force_only_DMP" "Force_only_Hist")

for env_idx in $(seq 0 $((num_exp_env - 1)))
do
    #sbatch -J "${names[$exp_idx]}_test" -a 1-$num_exp_per_env exp_control/hpc_batch.bash $env_idx $num_agents ppo
    sbatch -a 1-$num_exp_per_env exp_control/hpc_batch.bash $env_idx $num_agents "$1_$env_idx"
done