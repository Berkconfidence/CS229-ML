import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

X_train, y_train = load_data("data/ex2data1.txt")

for i in range(10):
    print(X_train[i]*2)
    for j in range(2):
        print(X_train[i,j]*2)