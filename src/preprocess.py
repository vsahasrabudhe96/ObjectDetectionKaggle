import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image

DATA_PATH = os.path.join("../data/")
targets = os.listdir(DATA_PATH)

def create_df(DATA_PATH):
    targets = os.listdir(DATA_PATH)
    X,y = [],[]
    for target in targets:
        for val in os.listdir(os.path.join(DATA_PATH,target)):
            X.append(os.path.join(DATA_PATH,target,val))
            y.append(target)
    df = pd.DataFrame(columns=['img_path','target'])
    df['img_path'] = X
    df['target'] = y
    return df


def reshape_images(df):
    return [cv2.resize(cv2.imread(i),(224,224))/255.0 for i in df['img_path']]


