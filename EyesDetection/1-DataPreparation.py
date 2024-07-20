import os
import shutil
import glob
from tqdm import tqdm

str = 's0001_00001_0_0_0_0_0_01.png'
str.split('_')

str.split('_')[4] # 0 means close eyes and 1 means open eyes


RAW_DIR = r'C:\Users\39329\Desktop\Progetto CV\mrlEyes_2018_01\mrlEyes_2018_01'
for dirpath, dirname, filenames in os.walk(RAW_DIR):
    for i in tqdm([f for f in filenames if f.endswith('.png')]):
        if i.split('_')[4] == '0':
            shutil.copy(src=dirpath+'/'+i, dst = r'C:\Users\39329\Desktop\Progetto CV\mrlEyes_2018_01\DDP\close_eyes')
        elif i.split('_')[4] == '1':
            shutil.copy(src=dirpath+'/'+i, dst = r'C:\Users\39329\Desktop\Progetto CV\mrlEyes_2018_01\DDP\open_eyes')



