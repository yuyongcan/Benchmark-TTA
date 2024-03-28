# [Benchmarking Test-Time Adaptation against Distribution Shifts in Image Classification](https://arxiv.org/abs/2307.03133)
## Prerequisites

To use the repository, we provide a conda environment.
```bash
conda update conda
conda env create -f environment.yaml
conda activate Benchmark_TTA 
```

## Structure of Project

This project contains several directories. Their roles are listed as follows:

+ ./best_cfgs: the best config files for each dataset and algorithm are saved here.
+ ./robustbench: a official library we used to load robust datasets and models. 
+ ./src/
  + data: we load our datasets and dataloaders by code under this directories.
  + methods: the code for implements of various TTA methods.
  + models: the various models' loading process and definition rely on the code here.
  + utils: some useful tools for our projects. 

## Run


This repository allows to study a wide range of different datasets, models, settings, and methods. A quick overview is given below:

- **Datasets**
  
  - `cifar10_c` [CIFAR10-C](https://zenodo.org/record/2535967#.ZBiI7NDMKUk)
  
  - `cifar100_c` [CIFAR100-C](https://zenodo.org/record/3555552#.ZBiJA9DMKUk)
  
  - `imagenet_c` [ImageNet-C](https://zenodo.org/record/2235448#.Yj2RO_co_mF)
  
  - `domainnet126` [DomainNet (cleaned)](http://ai.bu.edu/M3SDA/)
  
  - `officehome` [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?usp=sharing&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw)
  
  - The dataset directory structure is as follows:



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

  


  You can download the .txt file for DomainNet in ./dataset/DomainNet, generate .txt file for office-home following [SHOT](https://github.com/tim-learn/SHOT)

​	

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

download the ckpt of pretrained models and data load sequences from [here](https://drive.google.com/drive/folders/14GWvsEI5pDc3Mm7vqyELeBPuRUSPt-Ao?usp=sharing) and put it in ./ckpt
#### How to reproduce

The entry file for SHOT, NRC, PLUE to run is **SFDA-eva.sh**

To evaluate this methods, modify the DATASET and METHOD in SFDA-eva.sh

and then

```shell
bash SFDA-eva.sh
```

The entry file for other algorithms is **test-time-eva.sh**

 To evaluate this methods, modify the DATASET and METHOD in test-time-eva.sh

and then

```shell
bash test-time-eva.sh
```

## Add your own algorithm, dataset and model

We decouple the loading of datasets, models, and methods. So you can add them to our benchmarks completely independently.

### To add a algorithm

1. You can add a python files  **Algorithm_XX.py** for your algorithm in ./src/methods/

2. Add the setup process function of your algorithm **setup_XX(model, cfg)** in function ./src/methods/setup.py.

3. Add two line of your setup code in line 22 on ./test-time.py like

   ~~~python
       elif cfg.MODEL.ADAPTATION == "XX":
           model, param_names = setup_XX(base_model, cfg)
   ~~~

### To add a dataset

1. Write a function **load_dataset_name()** to load your dataset **Dataset_new** in ./src/data/data.py

2. Define the transforms used to load your dataset on function **get_transform()** in ./src/data/data.py

3. Add two line to load your dataset in function **load_dataset()** in ./src/data/data.py like

   ```python
       elif dataset == 'dataset_name':
           return load_dataset_name(root=root, batch_size=batch_size, workers=workers, split=split, transforms=transforms,
                                ckpt=ckpt)
   ```

### To add a model 

1. Just add the code for loading your model in **load_model()** function in ./src/model/load_model.py like

   ```python
       elif model_name == 'model_new':
           model =# the code for loading your model
   ```

   

## Acknowledgements

+ Robustbench [official](https://github.com/RobustBench/robustbench)
+ CoTTA [official](https://github.com/qinenergy/cotta)
+ TENT [official](https://github.com/DequanWang/tent)
+ AdaContrast [official](https://github.com/DianCh/AdaContrast)
+ EATA [official](https://github.com/mr-eggplant/EATA)
+ LAME [official](https://github.com/fiveai/LAME)
+ MEMO [official](https://github.com/zhangmarvin/memo)

