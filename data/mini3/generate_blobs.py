# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:15:49 2018

@author: wdrosko
"""

from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
import pandas as pd
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

save_prefix = "artificial_blobs_"
save_suffix = "_centers_v2.csv"

nums = [2,3,4,5,6,7,8,9,10,15,20,25,30]
for i in nums:
    save_name = save_prefix + str(i)+ save_suffix
    X, y = make_blobs(n_samples=3000, centers=i, n_features=5)
    
    rand_dat = [sum(i) for i in X]
    
    random_dat = DataFrame(rand_dat)
    class_data = DataFrame(y)
    clean_data = DataFrame(X)
    mu, sigma = 0, 0.6
    n_samples, n_features = clean_data.shape
    noise = np.random.normal(mu, sigma, [n_samples,n_features])
    signal = clean_data+noise
    out_data= pd.concat([signal,random_dat, class_data], axis=1)
    out_data.to_csv(save_name, index =False)
    

