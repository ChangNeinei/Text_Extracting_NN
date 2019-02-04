import numpy as np
import scipy.io
from nn import *
from collections import Counter
from skimage.util import random_noise
from skimage import data, img_as_float

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']
valid_label = valid_data['valid_labels']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()
plot_training_loss = []
plot_valid_loss = []

# initialize layers here
initialize_weights(1024, 32, params, 'layer1')
initialize_weights(32, 32, params, 'hidden')
initialize_weights(32, 32, params, 'hidden2')
initialize_weights(32, 1024, params, 'output')


# should look like your previous training loops

for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        h_1 = forward(xb, params, 'layer1', relu)
        h_2 = forward(h_1, params, 'hidden', relu)
        h_3 = forward(h_2, params, 'hidden2', relu)
        pred_x = forward(h_3, params, 'output',sigmoid)
        # loss
        loss = np.sum((xb - pred_x) ** 2)
        
        # backward
        delta1 = 2 * (pred_x - xb)
        delta = backwards(delta1, params, 'output', sigmoid_deriv) 
        delta_2 = backwards(delta, params,'hidden2', relu_deriv)
        delta_3 = backwards(delta_2, params,'hidden', relu_deriv)
        delta_4 = backwards(delta_3, params,'layer1', relu_deriv) 
        # apply gradient
        params['m_Woutput'] = 0.9 * params['m_Woutput'] - learning_rate * params['grad_Woutput']
        params['m_boutput'] = 0.9 * params['m_boutput'] - learning_rate * params['grad_boutput']
        params['Woutput'] = params['Woutput'] + params['m_Woutput'] 
        params['boutput'] = params['boutput'] + params['m_boutput']

        params['m_Whidden2'] = 0.9 * params['m_Whidden2'] - learning_rate * params['grad_Whidden2']
        params['m_bhidden2'] = 0.9 * params['m_bhidden2'] - learning_rate * params['grad_bhidden2']
        params['Whidden2'] = params['Whidden2'] + params['m_Whidden2'] 
        params['bhidden2'] = params['bhidden2'] + params['m_bhidden2']
        
        params['m_Whidden'] = 0.9 * params['m_Whidden'] - learning_rate * params['grad_Whidden']
        params['m_bhidden'] = 0.9 * params['m_bhidden'] - learning_rate * params['grad_bhidden']
        params['Whidden'] = params['Whidden'] + params['m_Whidden'] 
        params['bhidden'] = params['bhidden'] + params['m_bhidden']

        params['m_Wlayer1'] = 0.9 * params['m_Wlayer1'] - learning_rate * params['grad_Wlayer1']
        params['m_blayer1'] = 0.9 * params['m_blayer1'] - learning_rate * params['grad_blayer1']
        params['Wlayer1'] = params['Wlayer1'] + params['m_Wlayer1'] 
        params['blayer1'] = params['blayer1'] + params['m_blayer1']
        total_loss += loss
    plot_training_loss.append(total_loss)

    v_h1 = forward(valid_x,params,'layer1',relu)
    v_h2 = forward(v_h1,params,'hidden',relu)
    v_h3 = forward(v_h2,params,'hidden2',relu)
    v_out = forward(v_h3,params,'output',sigmoid)
    v_loss = np.sum((valid_x - v_out) ** 2)
    
    plot_valid_loss.append(v_loss)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# visualize some results
# Q5.3.1
import matplotlib.pyplot as plt

x_axis = np.arange(0, max_iters, 1)
plt.plot(x_axis, plot_training_loss, label = 'training loss', )
plt.plot(x_axis, plot_valid_loss, label = 'validation loss')
plt.legend(loc='upper right', shadow=True)
plt.title('Training and Validation Loss')
plt.xlabel('epoch number')
plt.ylabel('loss')
plt.show()


h1 = forward(valid_x,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid)


class_select = []
class_predict = []
for i in range(5):
    for index, label in enumerate(valid_label):
        if i == np.nonzero(label)[0]:
            class_select.append(valid_x[index])
            class_select.append(valid_x[index + 1])
            class_predict.append(out[index])
            class_predict.append(out[index + 1])
            break

for i in range(10):
    plt.subplot(2,1,1)
    plt.imshow(class_select[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(class_predict[i].reshape(32,32).T)
    plt.show()
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
# Q5.3.2
total_psnr = 0
for i in range(len(valid_x)):
    p = psnr(valid_x[i].reshape(32,32).T, out[i].reshape(32,32).T)
    total_psnr += p

print(total_psnr / len(valid_x))









