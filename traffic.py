import math
import pickle

import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
from scipy.misc import imresize


# TODO: fill this in based on where you saved the training and testing data
training_file = 'train.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,
    y_train,
    test_size=0.3,
    random_state=0)


### To start off let's do a basic data summary.

# TODO: number of training examples
n_train = len(X_train)

# TODO: number of testing examples
n_test = len(X_test)

# TODO: what's the shape of an image?
image_shape = 32*32*3

# TODO: how many classes are in the dataset
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
print("X_Train type", type(X_train[0]))

### Preprocess the data here.
### Feel free to use as many code cells as needed.
from sklearn.preprocessing import OneHotEncoder

def OHE_labels(Y_tr,N_classes):
    OHC = OneHotEncoder()

    Y_ohc = OHC.fit(np.arange(N_classes).reshape(-1, 1))
    Y_labels = Y_ohc.transform(Y_tr.reshape(-1, 1)).toarray()
    Y_labels.astype(np.float32)
    return Y_labels

def next_batch(x, y, batch_index, batch_size):
    start = batch_index + 1
    end = start + batch_size
    return x[start:end, :], y[start:end]

train_labels = OHE_labels(y_train,n_classes)
test_labels = OHE_labels(y_test,n_classes)
valid_labels = OHE_labels(y_valid,n_classes)

print("train label shape=",train_labels.shape)
print("test label shape=",test_labels.shape)
print("valid label shape=",valid_labels.shape)

layer_width = {
    'layer_1': 32,
    'layer_2': 64,
    'layer_3': 128,
    'fully_connected': 512
}

# Store layers weight & bias
weights = {
    'layer_1': tf.Variable(tf.truncated_normal([5, 5, 3, layer_width['layer_1']])),
    'layer_2': tf.Variable(tf.truncated_normal([5, 5, layer_width['layer_1'], layer_width['layer_2']])),
    'layer_3': tf.Variable(tf.truncated_normal([5, 5, layer_width['layer_2'], layer_width['layer_3']])),
    'fully_connected': tf.Variable(tf.truncated_normal([128, layer_width['fully_connected']])),
    'out': tf.Variable(tf.truncated_normal([layer_width['fully_connected'], n_classes]))
}
biases = {
    'layer_1': tf.Variable(tf.zeros(layer_width['layer_1'])),
    'layer_2': tf.Variable(tf.zeros(layer_width['layer_2'])),
    'layer_3': tf.Variable(tf.zeros(layer_width['layer_3'])),
    'fully_connected': tf.Variable(tf.zeros(layer_width['fully_connected'])),
    'out': tf.Variable(tf.zeros(n_classes))
}

def conv2d(x, W, b, strides=2):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.tanh(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')

# Create model
def conv_net(x, weights, biases):
    image = tf.reshape(x, shape=[-1, 32, 32, 3])

    # Layer 1
    conv1 = conv2d(image, weights['layer_1'], biases['layer_1'])
    conv1 = maxpool2d(conv1)

    # Layer 2
    conv2 = conv2d(conv1, weights['layer_2'], biases['layer_2'])
    conv2 = maxpool2d(conv2)

    # Layer 3
    conv3 = conv2d(conv2, weights['layer_3'], biases['layer_3'])
    conv3 = maxpool2d(conv3)

    # Fully connected layer
    # Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape( conv3, [-1, weights['fully_connected'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['fully_connected']), biases['fully_connected'])
    fc1 = tf.nn.tanh(fc1)

    # Output Layer - class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

learning_rate = 0.001
batch_size = 512
# training_epochs = 30
training_epochs = 1

# tf Graph input
x = tf.placeholder("float", [None, 32, 32, 3])
y = tf.placeholder("float", [None, n_classes])

# define all the variables and functions
logits = conv_net(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


# Initializing the variables
init = tf.initialize_all_variables()

print("Everything is defined")

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        batch_count = int(math.ceil(len(X_train)/batch_size))
        # The training cycle
        for i in range(batch_count):

            batch_x, batch_y = next_batch(X_train,train_labels,i,batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        # Display logs per epoch step
        c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(
        "Accuracy:",
        accuracy.eval({x: X_valid, y: valid_labels}))

def resize():
    images = os.listdir("webimages/")
    for f in images:
        #reading in an image
        image = mpimg.imread(os.path.join("webimages/", f))
        print('This image is:', type(image), 'with dimesions:', image.shape)
        resized= imresize(image,(32, 32))
        rows,cols,ch = image.shape
        # r = 32.0 / image.shape[1]
        # dim = (32, int(image.shape[0] * r))
        #
        # # perform the actual resizing of the image and show it
        # resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        print('resized:', resized.shape)
        cv2.imwrite(os.path.join("webresized",f), resized)

def image_feature():
    features = []

    images = os.listdir("webresized/")

    for f in images:
        # Check if the file is a directory
        image = mpimg.imread(os.path.join("webresized/", f))
        # # Load image data as 1 dimensional array
        # # We're using float32 to save on memory space
        feature = np.array(image, dtype=np.float32)
        # image = np.reshape(feature, (1, 32, 32, 3))
        print(feature.shape)
        features.append(feature)

    return np.array(features)

resize()
features = image_feature()

t1 = tf.placeholder("float", [32, 32, 3])

prob = tf.nn.softmax(t1)

with tf.Session() as sess:
    #print("softmax probabilities")
    for f1 in features:
        print(sess.run(prob, feed_dict={t1: f1}))


init_op = tf.initialize_all_variables()
prediction = tf.nn.softmax(logits)
topFive=tf.nn.top_k(prediction, k=5, sorted=True, name=None)
top_k_feed_dict = {x: features}

# Run the tf.nn.top_k operation in the session
with tf.Session() as session:
    session.run(init_op)
    top_k_probabilities_per_image = session.run(topFive, feed_dict=top_k_feed_dict)
    values = np.array([top_k_probabilities_per_image.values])
    indices = np.array([top_k_probabilities_per_image.indices])
    print(values)
    print(indices)

