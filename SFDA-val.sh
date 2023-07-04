DATASET="imagenet_c" # cifar10_c cifar100_c imagenet_c domainnet126 officehome
METHOD="nrc"          # shot nrc plue

echo DATASET: $DATASET
echo METHOD: $METHOD

GPUS=(1 5 6) #available gpus
NUM_GPUS=${#GPUS[@]}
NUM_MAX_JOB=$((NUM_GPUS))
i=0
#### Useful functions
wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=$NUM_MAX_JOB #num concurrent jobs
  local num_max_jobs=${1:-$default_num_jobs}
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}

test_time_adaptation() {
#  echo 0
  if [ "$METHOD" == "shot" ]; then
    lrs=(2e-4 5e-4 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2)
    if [ "$DATASET" == "imagenet_vit" ]; then
      lrs=(5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2)
    fi
    cls_pars=(0.1 0.2 0.3 0.5 0.9)
    for lr in ${lrs[*]}; do
      for cls_par in ${cls_pars[*]}; do
        wait_n
        i=$((i + 1))
        CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python SFDA-validation.py --cfg "cfgs/Online_TTA/${DATASET}/${METHOD}.yaml" --output_dir "SFDA-validation/${DATASET}/${METHOD}" \
          --OPTIM_LR "$lr" --SHOT_CLS_PAR "$cls_par" &
        sleep 10
      done
    done

  elif [ "$METHOD" == "nrc" ]; then
    lrs=(1e-4 1e-3 1e-2  5e-2)
    ks=(2 3 4 5)
    kks=(2 3 4 5)
    if [ "$DATASET" == "imagenet_vit" ]; then
      lrs=(5e-5 1e-4 2e-4 1e-3)
      ks=(2 3 5)
      kks=(2 3 5)
    elif [ "$DATASET" == "imagenet_c" ]; then
      lrs=(1e-5 5e-5 1e-4)
      ks=(3 4 5 )
      kks=(3 4 5)
    fi
    for lr in ${lrs[*]}; do
      for k in ${ks[*]}; do
        for kk in ${kks[*]}; do
          wait_n
          i=$((i + 1))
          CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python SFDA-validation.py --cfg "cfgs/Online_TTA/${DATASET}/${METHOD}.yaml" --output_dir "SFDA-validation/${DATASET}/${METHOD}" \
            --OPTIM_LR "$lr" --NRC_K "$k" --NRC_KK "$kk" &
          sleep 1
        done
      done
    done

  elif [ "$METHOD" == "plue" ]; then
    lrs=(5e-5 1e-4 1e-3 1e-2 5e-2)
    num_neighbors=(5 10 15)
#    echo 1
    for lr in ${lrs[*]}; do
      for num_neighbor in ${num_neighbors[*]}; do
#        echo 2
        wait_n
        i=$((i + 1))
#        echo 3
        CUDA_VISIBLE_DEVICES="${GPUS[i % ${NUM_GPUS}]}" python SFDA-validation.py --cfg "cfgs/Online_TTA/${DATASET}/${METHOD}.yaml" --output_dir "SFDA-validation/${DATASET}/${METHOD}" \
          --OPTIM_LR "$lr" --PLUE_NUM_NEIGHBORS "$num_neighbor" &
        sleep 100
      done
    done

  fi
}

test_time_adaptation
