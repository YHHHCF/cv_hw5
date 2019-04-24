import numpy as np
import scipy.io
from nn import *


def plotline(train_data, val_data, xlabel, ylabel, title):
    plt.plot(val_data[0], val_data[1], 'r', label='val')
    plt.plot(train_data[0], train_data[1], 'b', label='train')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(bbox_to_anchor=[0.3, 1])
    plt.grid()
    plt.show()


def show_param(param1, param2):
    fig = plt.figure(1, (4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, 2),  # creates 2x2 grid of axes
                     axes_pad=0.1, )  # pad between axes in inch.

    grid[0].imshow(param1)
    grid[1].imshow(param2)
    plt.show()
    return


train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 2000
# pick a batch size, learning rate
batch_size = 128
learning_rate = 1e-3
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(1024,64,params,'hidden')
initialize_weights(64,36,params,'output')

train_acc = [[], []]
val_acc = [[], []]
train_loss = [[], []]
val_loss = [[], []]

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    batches = get_random_batches(train_x, train_y, batch_size)
    total_loss = 0
    total_acc = 0
    cnt = 0

    for xb,yb in batches:
        # forward
        out = forward(xb, params, 'hidden')
        probs = forward(out, params, 'output', softmax)

        # loss
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += sum(loss)
        total_acc += acc
        cnt += 1

        # backward
        delta1 = probs - yb
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'hidden', sigmoid_deriv)

        # apply gradient
        grad_W_o = params['grad_W' + 'output']
        grad_b_o = params['grad_b' + 'output']

        grad_W_h = params['grad_W' + 'hidden']
        grad_b_h = params['grad_b' + 'hidden']

        params['W' + 'output'] -= learning_rate * grad_W_o.T
        params['b' + 'output'] -= learning_rate * grad_b_o
        params['W' + 'hidden'] -= learning_rate * grad_W_h.T
        params['b' + 'hidden'] -= learning_rate * grad_b_h

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss / cnt,total_acc / cnt))
        train_acc[0].append(itr)
        train_acc[1].append(total_acc / cnt)
        train_loss[0].append(itr)
        train_loss[1].append(total_loss / cnt)

    if itr % 20 == 0:
        batches = get_random_batches(valid_x, valid_y, batch_size)
        total_loss = 0
        total_acc = 0
        cnt = 0

        for xb, yb in batches:
            out = forward(xb, params, 'hidden')
            probs = forward(out, params, 'output', softmax)
            loss, acc = compute_loss_and_acc(yb, probs)

            total_loss += sum(loss)
            total_acc += acc
            cnt += 1

        print("val itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss / cnt, total_acc / cnt))
        val_acc[0].append(itr)
        val_acc[1].append(total_acc / cnt)
        val_loss[0].append(itr)
        val_loss[1].append(total_loss / cnt)

# run on validation set and report accuracy! should be above 75%
valid_acc = val_acc[1][-1]

plotline(train_acc, val_acc, 'iter', 'acc', 'train/val acc')
plotline(train_loss, val_loss, 'iter', 'loss', 'train/val loss')

print('Validation accuracy: ',valid_acc)
# if True: # view the data
#     for crop in xb:
#         import matplotlib.pyplot as plt
#         plt.imshow(crop.reshape(32,32).T)
#         plt.show()

import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

q3_weights_optim = pickle.load(open('q3_weights.pickle','rb'), encoding="ASCII")
Whidden_optim = q3_weights_optim['Whidden'].reshape(256, 256)
bhidden_optim = q3_weights_optim['bhidden'].reshape(8, 8)
Woutput_optim = q3_weights_optim['Woutput'].reshape(48, 48)
boutput_optim = q3_weights_optim['boutput'].reshape(6, 6)

Whidden_init = params['Whidden'].reshape(256, 256)
bhidden_init = params['bhidden'].reshape(8, 8)
Woutput_init = params['Woutput'].reshape(48, 48)
boutput_init = params['boutput'].reshape(6, 6)

show_param(Whidden_init, Whidden_optim)
show_param(bhidden_init, bhidden_optim)
show_param(Woutput_init, Woutput_optim)
show_param(boutput_init, boutput_optim)


# Q3.1.3
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
# load the model and make prediction

batches = get_random_batches(valid_x, valid_y, batch_size)

import string
letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
print(letters)

for xb, yb in batches:
    out = forward(xb, q3_weights_optim, 'hidden')
    probs = forward(out, q3_weights_optim, 'output', softmax)
    pred = np.argmax(probs, axis=1)
    label = np.argmax(yb, axis=1)
    for i in range(len(pred)):
        confusion_matrix[label[i]][pred[i]] += 1

# visualize confusion matrix
import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()

