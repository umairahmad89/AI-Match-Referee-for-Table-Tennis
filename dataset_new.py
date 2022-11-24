from pyparsing import originalTextFor
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from PIL import Image
import json
import os
from copy import deepcopy
from utils import resize_around_coords,normalized_markups, markups_preprocessing
import torchvision.transforms as transforms


# events = {"racket_hit":0, "bounce":1, "empty_event":2, "net": 3}


class TTDataset(Dataset):

    def __init__(self,root_dir, transform1, transform2, events):
        
        self.transform1 = transform1
        self.transform2 = transform2
        self.root = root_dir
        self.markups = markups_preprocessing(self.root)
        self.events = events
        length = self.__len__()
        print(length)

        indexes = np.random.choice(length,length)





    def __len__(self):

        length = 0
        for folder in os.listdir(self.root):
            
            folder_name = [name for name in os.listdir(os.path.join(self.root,folder))\
                 if not name.endswith(".json")][0]

            temp = len([file for file in os.listdir(os.path.join(\
                self.root,folder,folder_name))])
                
            length += temp

        return length

    def __getitem__(self, idx):

        image_path = list(self.markups.keys())[idx]
        markups = normalized_markups(self.markups[image_path],self.events)
        # now we have {"x":_, "y":_,"event":_}

        img = Image.open(image_path)

        if self.transform1:
            # in case of training, i need:
            # 1. downscalled image
            # 2. cropped image for local segment block
            # 3. markups
            # # 
            c_img = resize_around_coords(deepcopy(img), (markups["x"],markups["y"]))
            t_img = self.transform1(img)
            c_img = self.transform1(c_img)
            o_img = self.transform2(img)
            # we need to divide the markup into, coordinates and events.
            coordinates = torch.tensor(list(markups.values())[:2],dtype=torch.float32)
            event = torch.tensor(list(markups.values())[-1],dtype=torch.long)

        return t_img, o_img, c_img, coordinates,event


class Subset(TTDataset):

    def __init__(self, dataset, indices):
        
        self.dataset = dataset

        self.indices =  indices

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def __len__(self):
        return super().__len__()

