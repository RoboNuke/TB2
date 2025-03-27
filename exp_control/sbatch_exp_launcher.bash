#!/bin/bash

num_agents=4
num_exp_per_env=2
num_exp_env=5
names=("Std_Obs" "DMP_Obs" "Hist_Obs" "Force_only_DMP" "Force_only_Hist")
echo "started"
for env_idx in $(seq 0 $((num_exp_env - 1)))
do
    echo "${names[$exp_idx]}_test"
    sbatch -J "obs_test_${names[$exp_idx]}" -a=1-$num_exp_per_env hpc_batch.bash $env_idx $num_agents 
done