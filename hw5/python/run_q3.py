import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 8
learning_rate = 1e-2
hidden_size = 64

params = {}

print(train_x.shape, train_y.shape)
# initialize layers here
initialize_weights(1024, 64, params, 'layer1')
initialize_weights(64, 36, params, 'output')
assert (params['Wlayer1'].shape == (1024, 64))
assert (params['blayer1'].shape == (64,))

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

weights_init = params['Wlayer1']
fig1 = plt.figure()
grid = ImageGrid(fig1, 111, nrows_ncols=(8,8), axes_pad=0.0)
for i in range(weights_init.shape[1]):
    grid[i].imshow(weights_init[:,i].reshape((32, 32)))
plt.show()


# with default settings, you should get loss < 150 and accuracy > 80%
train_loss, train_acc = [], []
valid_loss, valid_acc = [], []
test_loss, test_acc = [], []
for itr in range(max_iters):

    batches_train = get_random_batches(train_x, train_y, batch_size)
    batch_num_train = len(batches_train)
    batches_valid = get_random_batches(valid_x, valid_y, batch_size)
    batch_num_valid = len(batches_valid)
    batches_test = get_random_batches(test_x, test_y, batch_size)
    batch_num_test = len(batches_test)

    total_loss = 0
    total_acc = 0
    for xb,yb in batches_train:
        # training loop can be exactly the same as q2!
        # forward
        post_act = forward(xb, params, 'layer1')
        probs = forward(post_act, params, 'output', softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc

        # backward
        # going in reverse direction
        delta1 = probs - yb
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)

        # apply gradient
        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        params['Woutput'] -= learning_rate * params['grad_Woutput']
        params['blayer1'] -= learning_rate * params['grad_blayer1']
        params['boutput'] -= learning_rate * params['grad_boutput']

    total_acc /= batch_num_train
    total_loss /= batch_num_train
    train_loss.append(total_loss)
    train_acc.append(total_acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t train loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

    # run on validation set and report accuracy! should be above 75%

    total_loss_v = 0
    total_acc_v = 0
    for xb, yb in batches_valid:
        # training loop can be exactly the same as q2!
        # forward
        post_act = forward(xb, params, 'layer1')
        probs = forward(post_act, params, 'output', softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss_v += loss
        total_acc_v += acc

    total_acc_v /= batch_num_valid
    total_loss_v /= batch_num_valid
    valid_loss.append(total_loss_v)
    valid_acc.append(total_acc_v)

    if itr % 2 == 0:
        print("itr: {:02d} \t valid loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss_v,total_acc_v))

    total_loss_t = 0
    total_acc_t = 0
    for xb, yb in batches_test:
        # training loop can be exactly the same as q2!
        # forward
        post_act = forward(xb, params, 'layer1')
        probs = forward(post_act, params, 'output', softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss_t += loss
        total_acc_t += acc

    total_acc_t /= batch_num_test
    total_loss_t /= batch_num_test
    test_loss.append(total_loss_t)
    test_acc.append(total_acc_t)

    if itr % 2 == 0:
        print("itr: {:02d} \t test loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss_t,total_acc_t))

print('Validation accuracy: ', valid_acc[-1])
print('Test accuracy: ', test_acc[-1])


plt.figure('Accuracy')
plt.plot(range(max_iters), train_acc, color='g')
plt.plot(range(max_iters), valid_acc, color='b')
# plt.plot(range(max_iters), test_acc, color='y')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Valid'])
plt.show()

plt.figure("Cross-Entropy Loss")
plt.plot(range(max_iters), train_loss, color='g')
plt.plot(range(max_iters), valid_loss, color='b')
# plt.plot(range(max_iters), test_loss, color='y')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.legend(['Train', 'Valid'])
plt.show()


if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
# visualize updated weights
weights_post = saved_params['Wlayer1']
fig1 = plt.figure()
grid = ImageGrid(fig1, 111, nrows_ncols=(8,8), axes_pad=0.0)
for i in range(weights_init.shape[1]):
    grid[i].imshow(weights_init[:, i].reshape((32, 32)))
plt.show()

# Q3.4
confusion_matrix = np.zeros((test_y.shape[1], test_y.shape[1]))

# compute confusion matrix here
h1 = forward(test_x, saved_params, 'layer1')
probs = forward(h1, saved_params, 'output', softmax)
test_true = np.argmax(test_y, axis=1)
test_pred = np.argmax(probs, axis=1)

for i in range(test_y.shape[0]):
    confusion_matrix[test_true[i], test_pred[i]] += 1

import string
plt.imshow(confusion_matrix, interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
