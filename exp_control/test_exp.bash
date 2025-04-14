names=( "Std_Obs" "Hist_Obs" )
envs=( "TB2-Factor-PiH-v0" "TB2-Factor-PiH-ObsHist-v0" )

exp_idx=$1
if [ "$exp_idx" -gt 4 ]; then
  exp_idx=0
fi

echo "Exp: ${names[$exp_idx]}"

#CUDA_LAUNCH_BLOCKING=1 
HYDRA_FULL_ERROR=1 python -m learning.single_agent_train \
    --task=${envs[$exp_idx]} \
    --max_steps=35000000 \
    --num_envs=256 \
    --num_agents=1 \
    --exp_name=$2  \
    --seed=1 \
    --headless \
    --log_smoothness_metrics \
    --learning_method="ppo" \
    --init_eval \
    --no_vids 

    #
    #
    #
    #
    #
    # 
    #
    #
   #
   # --no_log_wandb \
   # 
#
    #--wandb_tags="debug_Apr11" \#
    #
    #-
    #