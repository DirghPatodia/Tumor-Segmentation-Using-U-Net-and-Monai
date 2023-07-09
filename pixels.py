from monai.utils import first
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from monai.losses import DiceLoss
from tqdm import tqdm
from preprocess import prepare 

in_dir='D:/LungData/Final_data'
data_in= prepare(in_dir,cache=True)
train, test = data_in
def calculate_pixels1(train):
    val = np.zeros((1, 2))
   
    for batch in tqdm(train):
        batch_label = batch["seg"] != 0
        _, count = np.unique(batch_label, return_counts=True)

        if len(count) == 1:
            count = np.append(count, 0)
        val += count

    print('The last values:', val)
    return val

def calculate_pixels2(test):
    val = np.zeros((1, 2))
   
    for batch in tqdm(test):
        batch_label = batch["seg"] != 0
        _, count = np.unique(batch_label, return_counts=True)

        if len(count) == 1:
            count = np.append(count, 0)
        val += count

    print('The last values:', val)
    return val
print(calculate_pixels1(train))
print(calculate_pixels2(test))