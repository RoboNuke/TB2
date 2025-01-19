
names=("FPiH_wide_rel_IK" "FPiH_wide_jnt")
envs=("TB2-FPiH-Franka-Rel_IK-v0" ""TB2-FPiH-Franka-v0"")


for exp_idx in 0 1
do
    for seed in 1 2 3 4 5 
    do
        echo "${names[$exp_idx]}_$seed"
        python -m learning.single_agent_train --max_steps 25600000 \
            --headless --num_envs 256 \
            --eval_videos --train_videos \
            --wandb_project="TB2_Early_Tests" \
            --exp_name="${names[$exp_idx]}_$seed"  \
            --seed=$seed \
            --task=${envs[$exp_idx]}
    done
done