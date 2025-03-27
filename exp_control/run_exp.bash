
names=("Std_Obs" "DMP_Obs" "Hist_Obs" "Force_only_DMP" "Force_only_Hist")
envs=("TB2-Factor-PiH-v0" ) #"TB2-Factor-PiH-ObsDMP-v0" ) # "TB2-Factor-PiH-History-v0" )

# default num_agents is 2
if [ -z "$1" ]; then
    num_agents=2
else
    num_agents=$1
fi

for exp_idx in "${!envs[@]}"; 
do
    python -m learning.single_agent_train \
        --headless \
        --task=${envs[$exp_idx]} \
        --max_steps=50000000 \
        --no_vids \
        --num_envs=$((256 * $num_agents)) \
        --num_agents $num_agents \
        --exp_name="${names[$exp_idx]}"  \
        --wandb_project="DMP_Observation_Testing" \
        --wandb_tags="obs_tests" 
done

#python -m learning.single_agent_train --task TB2-Factor-PiH-v0 --exp_name basic_PiH_baseline --headless --max_steps 50000000 --no_vids --num_agents 5 --num_envs 1280 --wandb_tags multi_agent_tests basic_obs