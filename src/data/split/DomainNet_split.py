import os
from os.path import join as pjoin
from random import shuffle, seed
seed(2020)
DomainNet_path='/data2/yongcan.yu/datasets/DomainNet'
Domains=['clipart','painting','real','sketch']
for domain in Domains:
    all_path=pjoin(DomainNet_path,f'labeled_source_images_{domain}.txt')
    all_list=open(all_path,'r').readlines()
    shuffle(all_list)
    train_list=all_list[:int(len(all_list)*0.9)]
    val_list=all_list[int(len(all_list)*0.9):]
    train_path=pjoin(DomainNet_path,f'{domain}126_train.txt')
    test_path=pjoin(DomainNet_path,f'{domain}126_test.txt')
    open(train_path,'w').writelines(train_list)
    open(test_path,'w').writelines(val_list)
    print(f'{domain} train:{len(train_list)} test:{len(val_list)}')