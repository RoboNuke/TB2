#!/bin/bash
##SBATCH --array=1-2             # set up the array
##SBATCH -J SFPiH			    # name of job
#SBATCH -A virl-grp	            # name of my sponsored account, e.g. class or research group, NOT ONID!
#SBATCH -p dgxh			# name of partition or queue
#SBATCH --time=4-23:00:00        # time limit on job: 2 days, 12 hours, 30 minutes (default 12 hours)
##SBATCH -N 1                   # number of nodes (default 1)
#SBATCH --gres=gpu:1            # number of GPUs to request (default 0)
#SBATCH --mem=32G               # request 10 gigabytes memory (per node, default depends on node)
#SBATCH -c 6                   # number of cores/threads per task (default 1)
##SBATCH --constraint=h100,v10
#SBATCH -o ../outs/SFPiH_%A_%a.out		# name of output file for this submission script
#SBATCH -e ../outs/SFPiH_%A_%a.err		# name of error file for this submission script
# load any software environment module required for app (e.g. matlab, gcc, cuda)

echo "Job Name:" $SLURM_JOB_NAME
echo "Array:" $SLURM_ARRAY_TASK_COUNT
##module load cuda/10.1
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate isaaclab

bash "exp_control/run_exp.bash" $1 $2