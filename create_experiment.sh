module load compiler/llvm
module load devel/python/3.11.7
source .venv/bin/activate

mlflow experiments create -n $1
