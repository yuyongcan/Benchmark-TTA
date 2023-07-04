DATASET="imagenet_vit" # cifar10_c cifar100_c imagenet_c domainnet126 officehome
METHOD="nrc"        # source norm_test memo eata cotta tent t3a norm_alpha lame adacontrast
#sleep 9700
echo DATASET: $DATASET
echo METHOD: $METHOD

GPU_id=1

CUDA_VISIBLE_DEVICES="$GPU_id" python SFDA.py --cfg best_cfgs/Online_TTA/${DATASET}/${METHOD}.yaml --output_dir "SFDA-evaluation/${DATASET}/${METHOD}" &
