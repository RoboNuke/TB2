
names=("Std_Obs" "DMP_Obs" "Hist_Obs" "Force_only_DMP" "Force_only_Hist")
envs=("TB2-Factor-PiH-v0" "TB2-Factor-PiH-ObsDMP-v0" "TB2-Factor-PiH-History-v0" "TB2-Factor-PiH-ObsDMP-ForceOnly-v0" "TB2-Factor-PiH-History-ForceOnly-v0" )

# default num_agents is 2
if [ -z "$2" ]; then
    num_agents=2
else
    num_agents=$2
fi

exp_idx=$1
if [ "$exp_idx" -gt 4 ]; then
  exp_idx=0
fi
echo "Exp: ${names[$exp_idx]}"
echo "Num Agents: $num_agents"
python -m learning.single_agent_train \
    --headless \
    --task=${envs[$exp_idx]} \
    --max_steps=50000000 \
    --no_vids \
    --num_envs=$((256 * $num_agents)) \
    --num_agents $num_agents \
    --exp_name="${names[$exp_idx]}"  \
    --wandb_project="DMP_Observation_Testing" \
    --wandb_tags="obs_tests" \
    --seed=1 \
    --no_log_wandb \
    --log_smoothness_metrics

#python -m learning.single_agent_train --task TB2-Factor-PiH-v0 --exp_name basic_PiH_baseline --headless --max_steps 50000000 --no_vids --num_agents 5 --num_envs 1280 --wandb_tags multi_agent_tests basic_obs