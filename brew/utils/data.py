import numpy as np
import sklearn.cross_validation



def split_data(X, y, t_size):
    if len(X) != len(y):
        return None

    if hasattr(sklearn.cross_validation, 'train_test_split'):
        return sklearn.cross_validation.train_test_split(X, y, test_size=t_size)
    
    return None
        
    

