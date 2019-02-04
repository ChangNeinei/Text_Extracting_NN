import numpy as np
import scipy.io
from mpl_toolkits.axes_grid1 import ImageGrid
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

# train_x 
max_iters = 80
# pick a batch size, learning rate
batch_size = 60
learning_rate = 1e-3
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}
plot_training_loss = []
plot_valid_loss = []
plot_training_acc = []
plot_valid_acc = []

# initialize layers here
initialize_weights(1024, hidden_size, params, 'layer1')
initialize_weights(hidden_size, 36, params, 'output')
init_W = params['Wlayer1']

# with default settings, you should get loss < 150 and accuracy > 80%
# run on validation set and report accuracy! should be above 75%

for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    valid_acc = 0
    for xb,yb in batches:
        h_1 = forward(xb, params, 'layer1')
        pred_y = forward(h_1, params, 'output',softmax)
        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, pred_y)
        # backward
        delta1 = pred_y - yb
        
        delta = backwards(delta1, params, 'output', linear_deriv) #[5 * 25]
        delta_2 = backwards(delta, params,'layer1', sigmoid_deriv) #[5 * 2]
        # apply gradient
        W_1 = params['W' + 'output'] # [25 * 4]
        b_1 = params['b' + 'output'] # [4]
        del_w_1 = learning_rate * params['grad_W' + 'output']
        del_b_1 = learning_rate * params['grad_b' + 'output']
        params['W' + 'output'] = W_1 - del_w_1
        params['b' + 'output'] = b_1 - del_b_1
        W_2 = params['W' + 'layer1']
        b_2 = params['b' + 'layer1']
        del_w_2 = learning_rate * params['grad_W' + 'layer1']
        del_b_2 = learning_rate * params['grad_b' + 'layer1']
        params['W' + 'layer1'] = W_2 - del_w_2
        params['b' + 'layer1'] = b_2 - del_b_2
        total_loss += loss
        total_acc += acc
    total_loss = total_loss / len(batches)
    plot_training_loss.append(total_loss)
    total_acc = total_acc / len(batches)
    plot_training_acc.append(total_acc)

    valid_h_1 = forward(valid_x, params, 'layer1')
    valid_predict_y = forward(valid_h_1, params, 'output',softmax)
    valid_loss, valid_acc = compute_loss_and_acc(valid_y, valid_predict_y)
    valid_loss = valid_loss / len(batches)
    plot_valid_loss.append(valid_loss)
    plot_valid_acc.append(valid_acc)
      
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

scipy.io.savemat('weight.mat',{'W_Output': params['Woutput'], 'b_Output': params['boutput'], 'W_layer1': params['Wlayer1'], 'b_layer1': params['Wlayer1']})  
print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

x_axis = np.arange(0, max_iters, 1)
plt.plot(x_axis, plot_training_loss, label = 'training loss', )
plt.plot(x_axis, plot_valid_loss, label = 'validation loss')
plt.legend(loc='upper right', shadow=True)

plt.title('Training and Validation Loss')
plt.xlabel('epoch number')
plt.ylabel('loss')
plt.show()

plt.plot(x_axis, plot_training_acc, label = 'training accuracy', )
plt.plot(x_axis, plot_valid_acc, label = 'validation accuracy')
plt.legend(loc='best', shadow=True)

plt.title('Training and Validation Accuracy')
plt.xlabel('epoch number')
plt.ylabel('accuracy')
plt.show()



# Q3.1.3
train_W = params['Wlayer1']
fig = plt.figure()
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )


for i in range(init_W.shape[1]):
    im = init_W[:,i].reshape(32,32)
    grid[i].imshow(im)  # The AxesGrid object work as a list of axes.

plt.show()

'''
for i in range(train_W.shape[1]):
    im = train_W[:,i].reshape(32,32)
    grid[i].imshow(im)  # The AxesGrid object work as a list of axes.

plt.show()


confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
test_h_1 = forward(test_x, params, 'layer1')
pred_test_y = forward(test_h_1, params, 'output',softmax)
max_output = np.amax(pred_test_y, axis = 1, keepdims = True)
pred_class = np.ones(test_y.shape) * (pred_test_y//max_output == 1)
true = np.nonzero(test_y)[1]
pred = np.nonzero(pred_class)[1]

test_size = test_x.shape[0]
for i in range(test_size):
    x_ind = pred[i]
    y_ind = true[i]
    confusion_matrix[y_ind][x_ind] = confusion_matrix[y_ind][x_ind] + 1



import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
'''