#!/bin/bash
# 
# name of the job for better recognizing it in the queue overview
#SBATCH --job-name=fourier_fingerprints
# 
# define how many nodes we need
#SBATCH --nodes=1
#
# we only need on 1 cpu at a time
#SBATCH --ntasks=10
#
# expected duration of the job
#              hh:mm:ss
#SBATCH --time=08:00:00
# 
# partition the job will run on
#SBATCH --partition cpu
# 
# expected memory requirements
#SBATCH --mem=16000MB
#
# infos
#
# output path
#SBATCH --output="logs/slurm/slurm-%j-%x.out"

module load compiler/llvm
module load devel/python/3.11.7

# ~/fourier_fingerprints/.venv/bin/python -m kedro run --pipeline coefficients --params=$1
# ~/fourier_fingerprints/.venv/bin/python -m kedro run --pipeline expressibility --params=$1

# For Coefficients and Expressibility
# for seed in 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009
# do
#     echo "--- Seed $seed ---"
#     ~/fourier_fingerprints/.venv/bin/python -m kedro run --pipeline coefficients --params="$1,seed=$seed"
#     # ~/fourier_fingerprints/.venv/bin/python -m kedro run --pipeline expressibility --params="$1,seed=$seed"
# done

# For Training
for training_seed in 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009
do
    echo "--- Training Seed $training_seed ---"
    ~/fourier_fingerprints/.venv/bin/python -m kedro run --pipeline training_fourier --params="$1,data.seed=$training_seed" &
    # ~/fourier_fingerprints/.venv/bin/python -m kedro run --pipeline training_hep --params="$1,data.seed=$training_seed"
    # ~/fourier_fingerprints/.venv/bin/python -m kedro run --pipeline training_classical --params="$1,data.seed=$training_seed"

    sleep 60
done
wait

# for encoding in RX RY RZ
# do
#     echo "Seed $seed"
#     ~/fourier_fingerprints/.venv/bin/python -m kedro run --pipeline coefficients --params="$1,model.encoding=$encoding"
# done

# Done
exit 0


