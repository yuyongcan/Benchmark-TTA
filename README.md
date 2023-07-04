# Benchmarking Test-Time Adaptation against Distribution Shifts in Image Classification

## Prerequisites
To use the repository, we provide a conda environment.
```bash
conda update conda
conda env create -f environment.yaml
conda activate Benchmark_TTA 
```

## Classification

<details open>
<summary>Features</summary>

This repository allows to study a wide range of different datasets, models, settings, and methods. A quick overview is given below:

- **Datasets**
  
  - `cifar10_c` [CIFAR10-C](https://zenodo.org/record/2535967#.ZBiI7NDMKUk)
  
  - `cifar100_c` [CIFAR100-C](https://zenodo.org/record/3555552#.ZBiJA9DMKUk)
  
  - `imagenet_c` [ImageNet-C](https://zenodo.org/record/2235448#.Yj2RO_co_mF)
  
  - `domainnet126` [DomainNet (cleaned)](http://ai.bu.edu/M3SDA/)
  
  - `officehome` [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?usp=sharing&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw)
  
  - The dataset directory structure is as follows:
  - 
        |-- datasets
  
            |-- cifar-10
  
            |-- cifar-100
  
            |-- ImageNet
  
                |-- train
  
                |-- val
  
            |-- ImageNet-C
  
            |-- CIFAR-10-C
  
            |-- CIFAR-100-C
  
            |-- DomainNet
  
                |-- clipart
  
                |-- painting
  
                |-- real
  
                |-- sketch
  
                | -- clipart126_test.txt
  
                ......
  
            |-- office-home
  
                |-- Art
  
                |-- Clipart
  
                |-- Product
  
                |-- Real_World
  
  
  
  You can download the .txt file for DomainNet in ./dataset/DomainNet, we generate .txt file for office-home following [SHOT](https://github.com/tim-learn/SHOT)

  
- **Models**
  
  - For adapting to ImageNet variations, ResNet-50 models available in [Torchvision](https://pytorch.org/vision/0.14/models.html) can be used and ViT available in [timm · PyPI](https://pypi.org/project/timm/#models).
  - For the corruption benchmarks, pre-trained models from [RobustBench](https://github.com/RobustBench/robustbench) can be used.
  - For the DomainNet-126 benchmark, there is a pre-trained model for each domain.
  - The checkpoint of pretrained models is in directory ckpt

- **Methods**
  - The repository currently supports the following methods: source, [PredBN](https://arxiv.org/abs/2006.10963), [PredBN+](https://proceedings.neurips.cc/paper/2020/hash/85690f81aadc1749175c187784afc9ee-Abstract.html), [TENT](https://openreview.net/pdf?id=uXl3bZLkr3c),
    [MEMO](https://openreview.net/pdf?id=vn74m_tWu8O),  [EATA](https://arxiv.org/abs/2204.02610),
    [CoTTA](https://arxiv.org/abs/2203.13591), [AdaContrast](https://arxiv.org/abs/2204.10377), [LAME](https://arxiv.org/abs/2201.05718), [SHOT](https://arxiv.org/abs/2002.08546), [NRC](https://proceedings.neurips.cc/paper/2021/hash/f5deaeeae1538fb6c45901d524ee2f98-Abstract.html), [PLUE](https://arxiv.org/abs/2303.03770), [T3A](https://openreview.net/forum?id=e_yvNqkJKAW), [SAR](https://openreview.net/forum?id=g2YraF75Tj)


- **Modular Design**
  - Adding new methods should be rather simple, thanks to the modular design.

### Get Started
To run one of the following benchmarks, the corresponding datasets need to be downloaded.

Next, specify the root folder for all datasets `_C.DATA_DIR = "./data"` in the file `conf.py`. 

The best parameters for each method and dataset are save in ./best_cfgs

#### How to reproduce

The entry file for SHOT, NRC, PLUE to run is **SFDA-eva.sh**

To evaluate this methods, modify the DATASET and METHOD in SFDA-eva.sh

and then

```shell
bash SFDA-eva.sh
```


### Acknowledgements
+ Robustbench [official](https://github.com/RobustBench/robustbench)
+ CoTTA [official](https://github.com/qinenergy/cotta)
+ TENT [official](https://github.com/DequanWang/tent)
+ AdaContrast [official](https://github.com/DianCh/AdaContrast)
+ EATA [official](https://github.com/mr-eggplant/EATA)
+ LAME [official](https://github.com/fiveai/LAME)
+ MEMO [official](https://github.com/zhangmarvin/memo)

