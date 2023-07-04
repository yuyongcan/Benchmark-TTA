DATASET="domainnet126"     # cifar10_c cifar100_c imagenet_c domainnet126 officehome imagenet_vit
METHOD="sar"        # source norm_test memo eata cotta tent t3a norm_alpha lame adacontrast sar

echo DATASET: $DATASET
echo METHOD: $METHOD

GPU_id=3

CUDA_VISIBLE_DEVICES="$GPU_id" python test-time.py --cfg best_cfgs/Online_TTA/${DATASET}/${METHOD}.yaml --output_dir "test-time-evaluation/${DATASET}/${METHOD}" &