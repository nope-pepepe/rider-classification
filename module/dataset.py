#!/usr/bin/python
# -*- Coding: utf-8 -*-

import torch
import torch.utils.data
import csv
import os
import cv2
import numpy as np
from PIL import Image

class RiderDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir="rider-dataset", transform=None, size=(224, 224)):
        """
        csv_file : csvまでのパス
        root_dir : 画像までのパス
        transform: 変換方法
        size     : 変換後サイズ
        """
        print("start loading dataset")
        csvfile = open(csv_file, mode="r")
        self.transform = transform

        self.img_data = []
        self.label_data = []

        for data in csv.reader(csvfile):
            print(data)
            img = Image.open(os.path.join(root_dir, data[0]))
            img = img.resize(size)
            self.img_data.append(img)
            self.label_data.append(data[1])

        csvfile.close()
        print("loaded {} images!".format(len(self.label_data)))

    def __len__(self):
        return len(self.label_data)
    
    def __getitem__(self, idx):
        img = self.img_data[idx]
        label = self.label_data[idx]

        if self.transform:
            img = self.transform(img)
        
        return img, label

if __name__ == "__main__":
    dataset = RiderDataset("/home/kataoka/exthd/Kondo/github/rider-classification/rider-dataset/splits/train_1.csv")