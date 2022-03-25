import cv2 
import torch 
from torch import sigmoid as Sigmoid 
from torch import randn as Normal
from torch.nn.functional import interpolate as Interpolate 

import os
import requests
from random import randint 




class Sampler():
    def __init__(self):
        self.URLS = ['https://static.dw.com/image/42881723_401.jpg', 'https://static.dw.com/image/42881685_401.jpg', 
        'https://gdb.rferl.org/a46425b7-062b-4f1e-9e80-1dd345a90dfc_w408_r1_s.jpg', 'https://static.dw.com/image/42881766_303.jpg',
        'https://georgianjournal.ge/media/_thumb/images/georgianews/2015/June/World/zhirinovsky.jpg']
    
    def download_sample(self):
        exists = os.path.exists('the_third_pig.jpg')
        if exists:
            os.remove('the_third_pig.jpg')

        URL = self.URLS[randint(0, len(self.URLS)-1)]
        metadata = requests.get(URL).content
        with open('the_third_pig.jpg', 'wb') as handler:
            handler.write(metadata)
        
    def sample(self):
        self.download_sample()
        
        image = cv2.imread('the_third_pig.jpg')
        input = torch.from_numpy(image).type(torch.float32).permute(2,0,1).unsqueeze(0)
        resized_input = Interpolate(input, size=(224, 224))
        normal_distr = Normal(resized_input.shape, dtype=torch.float32)
        resized_input = (normal_distr + resized_input) / 255.
        return resized_input