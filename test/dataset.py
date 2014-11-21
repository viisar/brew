import numpy as np
import sklearn.datasets as datasets


# this indices will always be used so that we get reproduceable results in the tests
iris_index = np.array([ 69,  63,  32, 131,  13,  94,  10,  17,   4, 108,  29,  96, 100,
                       143,  20,  86,  35, 144,  78,  18,  11,  33,  72, 106,  24,  84,
                        42, 126,  51,  50,  90,  30, 146, 119,   1,  43,  37,  64,   5,
                       116, 122,  81,  45,  34, 112,  49,  31, 127, 114, 113,  41, 107,
                        22,  48, 137,  88, 110,  65, 105, 101,  23,  83,  26,  25, 111,
                        60,  68, 135, 109,   9,  47, 148, 142,  97, 130, 129,   6,  87,
                        58, 138,  73, 117, 133, 128,  39,  56,  85,  76, 104, 102,  38,
                        61,  92, 140,  70, 120,  12,  57, 134, 115, 147, 103,  82,  53,
                        62,  46, 118,  59,  36, 141, 132,  54,  44,  21,   7, 123, 125,
                       145,  99,  98,  79,  14, 139,  89,  74,  77,  66,  27, 149,   8,
                         0,  19,  95, 121,  16,  28,  40,  93, 124,  15,  52,  80, 136,
                        91,  67,   3,  75,   2,  55,  71])

def load_iris():
    iris = datasets.load_iris()

    data = iris['data']
    target = iris['target']
        
    # use fixed shuffling
    dataset = np.concatenate((data, target.reshape((150,1))), axis=1)[iris_index, :]

    # will always obtain the same train and test set 
    train_set = dataset[:105, :]    # 70%
    test_set = dataset[105:, :]  # 30%

    print(train_set.shape)
    print(test_set.shape)

