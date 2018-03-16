from data_utils import load_CIFAR10
import numpy as np
import random
import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

filename = '/Users/lvbaiyang/Desktop/tutorial/datasets/cifar-10-batches-py'
x_train, y_train, x_test, y_test = load_CIFAR10(filename)

# print 'Training data shape: ', x_train.shape
# print 'Training labels shape: ', y_train.shape
# print 'Test data shape: ', x_test.shape
# print 'Test labels shape: ', y_test.shape

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, classname in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)#find all the pictures in the same class
    idxs = np.random.choice(idxs, samples_per_class, replace=False)#choose 7 different pictures randomly
    # for i, idx in enumerate(idxs):
    #     plt_idx = i * num_classes + y + 1
    #     plt.subplot(samples_per_class, num_classes, plt_idx)
    #     plt.imshow(x_train[idx].astype('uint8'))
    #     plt.axis('off')
    #     if i == 0:
    #         plt.title(classname)

num_training = 5000
mask = list(range(num_training))
x_train = x_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
x_test = x_test[mask]
y_test = y_test[mask]

x_train = np.reshape(x_train, (x_train.shape[0], -1))#reshape the image data into rows
x_test = np.reshape(x_test, (x_test.shape[0], -1))

from classifiers import KNearestNeighbor

classifier = KNearestNeighbor()
classifier.train(x_train, y_train)
dists = classifier.compute_distances_one_loop(x_test)

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
x_train_folds = []
y_train_folds = []

x_train_folds = np.array_split(x_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies = {}
for i in k_choices:
    k_to_accuracies[i] = []



# y_test_pred = classifier.predict_labels(dists, k = 5)
# num_correct = np.sum(y_test == y_test_pred)
# accuracy = float(num_correct) / num_test
#
# print ('Got %d / %d correct => accuracy = %f' % (num_correct, num_test, accuracy))
#




















