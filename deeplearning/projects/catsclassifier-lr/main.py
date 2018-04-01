import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_x, train_y, test_x, test_y, classes = load_dataset()

index = 22
plt.imshow(train_x[index])
plt.show()
print ("y = " + str(train_y[:, index]) + ", it's a '" + classes[np.squeeze(train_y[:, index])]
       .decode("utf-8") +  "' picture.")

m_train = train_x.shape[0]
m_test = test_x.shape[0]
num_px = train_x.shape[1]

train_x_flatten = train_x.reshape(train_x.shape[0], -1).T
test_x_flatten = test_x.reshape(test_x.shape[0], -1).T
