#!/bin/bash
# 
# name of the job for better recognizing it in the queue overview
#SBATCH --job-name=entangling-the-waves
# 
# define how many nodes we need
#SBATCH --nodes=1
#
# we only need on 1 cpu at a time
#SBATCH --ntasks=6
#
# expected duration of the job
#              hh:mm:ss
#SBATCH --time=48:00:00
# 
# partition the job will run on
#SBATCH --partition cpu
# 
# expected memory requirements
#SBATCH --mem=10000MB
#
# infos
#
# output path
#SBATCH --output="logs/slurm/slurm-%j-%x.out"

module load compiler/llvm
module load devel/python/3.11.7

# ~/entangling-the-waves/.venv/bin/python -m kedro run --pipeline coefficients --params=$1
# ~/entangling-the-waves/.venv/bin/python -m kedro run --pipeline expressibility --params=$1

# for seed in 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009
# do
#     echo "--- Seed $seed ---"
#     ~/entangling-the-waves/.venv/bin/python -m kedro run --pipeline coefficients --params="$1,seed=$seed"
#     # ~/entangling-the-waves/.venv/bin/python -m kedro run --pipeline expressibility --params="$1,seed=$seed"
# done

for training_seed in 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009
do
    echo "--- Training Seed $training_seed ---"
    # ~/entangling-the-waves/.venv/bin/python -m kedro run --pipeline training_fourier --params="$1,data.seed=$training_seed"
    ~/entangling-the-waves/.venv/bin/python -m kedro run --pipeline training_hep --params="$1,data.seed=$training_seed"
done

# for encoding in RX RY RZ
# do
#     echo "Seed $seed"
#     ~/entangling-the-waves/.venv/bin/python -m kedro run --pipeline coefficients --params="$1,model.encoding=$encoding"
# done

# Done
exit 0


