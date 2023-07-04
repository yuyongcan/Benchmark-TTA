GPUS=(0 1 2 3 4 5) #available gpus
NUM_GPUS=${#GPUS[@]}
echo "$NUM_GPUS"
NUM_MAX_JOB=$(($NUM_GPUS * 2))
echo $NUM_MAX_JOB