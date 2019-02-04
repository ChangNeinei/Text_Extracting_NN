import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr
from scipy import linalg


train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']
valid_label = valid_data['valid_labels']
test_x = test_data['test_data']

dim = 32
# do PCA
print(train_x.shape)
meanx = np.mean(train_x, axis=1, keepdims=True)
shift_x = train_x - meanx
cov = (np.dot(shift_x.T, shift_x)) / len(train_x)
U,sigma,V = linalg.svd(cov)
project_matrix = V[:dim] #[32 * 1024]
print('Rank:' + str(np.linalg.matrix_rank(project_matrix)))


# rebuild a low-rank version
lrank = None
lrank = np.dot(shift_x, project_matrix.T) # n * 32
# rebuild it
recon = None
recon = np.dot(lrank, project_matrix)
recon = recon + meanx

for i in range(5):
    plt.subplot(2,1,1)
    plt.imshow(train_x[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(recon[i].reshape(32,32).T)
    plt.show()

# build valid dataset
mean_valid_x = np.mean(valid_x, axis=1, keepdims=True)
shift_valid_x = valid_x - mean_valid_x
valid_cov = (np.dot(shift_valid_x.T, shift_valid_x)) / len(valid_x)
U,sigma,V = linalg.svd(valid_cov)
project_valid_matrix = V[:32] #[32 * 1024]

recon_valid = np.dot(shift_valid_x, project_valid_matrix.T)
recon_valid = np.dot(recon_valid, project_valid_matrix)
recon_valid = recon_valid + mean_valid_x

class_select = []
class_recon = []

for i in range(5):
	for index, label in enumerate(valid_label):
		if i == np.nonzero(label)[0]:
			class_select.append(valid_x[index])
			class_select.append(valid_x[index + 1])
			class_recon.append(recon_valid[index])
			class_recon.append(recon_valid[index + 1])
			break

for i in range(10):
    plt.subplot(2,1,1)
    plt.imshow(class_select[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(class_recon[i].reshape(32,32).T)
    plt.show()

total = []
for pred,gt in zip(recon_valid,valid_x):
    total.append(psnr(gt,pred))
print(np.array(total).mean())






