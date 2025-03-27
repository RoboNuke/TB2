#!/bin/bash
SESSION_NAME="exp_testing"
gpu_path="exp_control/run_exp.bash"

tmux new-session -d -s "$SESSION_NAME"
tmux new-window -t "$SESSION_NAME":$i 
tmux send-keys -t "$SESSION_NAME":$i "conda activate isaaclab && "
tmux send-keys -t "$SESSION_NAME":$i "bash $gpu_path " 
tmux send-keys -t "$SESSION_NAME":$i "$*"
tmux send-keys -t "$SESSION_NAME":$i "; sleep 30; exit" C-m

window_count=$(tmux list-windows | wc -l)
while [ $window_count -gt 0 ]; do
    sleep 600
    window_count=$(tmux list-windows | wc -l)
    if [ "$window_count" -eq 1 ]; then
        # Close the tmux session
        window_count=0
        tmux kill-session
    fi
done
#tmux attach -t "$SESSION_NAME"