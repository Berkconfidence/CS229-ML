import numpy as np
import os

def load_data():
    # Dosyanın bulunduğu dizini al
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data/ex1data1.txt")
    data = np.loadtxt(data_path, delimiter=',')
    X = data[:,0]
    y = data[:,1]
    return X, y

def load_data_multi():
    # Dosyanın bulunduğu dizini al
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data/ex1data2.txt")
    data = np.loadtxt(data_path, delimiter=',')
    X = data[:,:2]
    y = data[:,2]
    return X, y