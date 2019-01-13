from dataloader import Dataloader
import os
import json
from random import shuffle
import numpy as np 
import tensorflow as tf 
import matplotlib.image as mpimg
from time import gmtime, strftime

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def L2_reg(weights):
    reg = tf.nn.l2_loss(weights)
    return reg


def conv_net(x, weights, biases, dropout):
    # Uncomment if you want to start with an FC for the input
    x = tf.contrib.layers.flatten(x)
    x = tf.contrib.layers.fully_connected(x, 16384, tf.nn.relu)
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 128, 128, 1]) 
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2) 
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    # 3rd Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    # Fully connected layer
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    # Add another FC layer
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # Apply Dropout
    fc2 = tf.nn.dropout(fc2, dropout)
    # Add another FC layer
    fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
    fc3 = tf.nn.relu(fc3)
    # Apply Dropout
    fc3 = tf.nn.dropout(fc3, dropout)
    # Add another FC layer
    fc4 = tf.add(tf.matmul(fc3, weights['wd4']), biases['bd4'])
    fc4 = tf.nn.relu(fc4)
    # Apply Dropout
    fc4 = tf.nn.dropout(fc4, dropout)
    # Add another FC layer
    fc5 = tf.add(tf.matmul(fc4, weights['wd5']), biases['bd5'])
    fc5 = tf.nn.relu(fc5)
    # Apply Dropout
    fc5 = tf.nn.dropout(fc5, dropout)
    # Add another FC layer
    fc6 = tf.add(tf.matmul(fc5, weights['wd6']), biases['bd6'])
    fc6 = tf.nn.relu(fc6)
    # # Apply Dropout
    fc6 = tf.nn.dropout(fc6, dropout)
    # # Add another FC layer
    # fc7 = tf.add(tf.matmul(fc6, weights['wd7']), biases['bd7'])
    # fc7 = tf.nn.relu(fc7)
    # # Apply Dropout
    # fc7 = tf.nn.dropout(fc7, dropout)
    # # Add another FC layer
    # fc8 = tf.add(tf.matmul(fc7, weights['wd8']), biases['bd8'])
    # fc8 = tf.nn.relu(fc8)
    # # Apply Dropout
    # fc8 = tf.nn.dropout(fc8, dropout)
    # # Add another FC layer
    # fc9 = tf.add(tf.matmul(fc8, weights['wd9']), biases['bd9'])
    # fc9 = tf.nn.relu(fc9)
    # # Apply Dropout
    # fc9 = tf.nn.dropout(fc9, dropout)
    # Add another FC layer
    # fc10 = tf.add(tf.matmul(fc9, weights['wd10']), biases['bd10'])
    # fc10 = tf.nn.relu(fc10)
    # # Apply Dropout
    # fc10 = tf.nn.dropout(fc10, dropout)
    # Output, class prediction
    out = tf.add(tf.matmul(fc6, weights['out']), biases['out'])
    return out

def batch_generator(loader, file_dir, table_name, batch_size=10):
    batches = []
    labels = []
    files = os.listdir(file_dir)
    shuffle(files)
    while True: 
        shuffle(files)
        for f in files:
            if f.endswith('.png'):
                data = loader.load_data(f, table_name)
                boxes = loader.grab_boxes(data)
                try:
                    X = [item[0] for item in boxes]
                    Y = [item[1][0] for item in boxes]
                    X = X[0]
                    Y = Y[0]
                except Exception:
                    print('Error getting data from file: ', f)
                    continue
                batches.append(X)
                labels.append(Y)
                if len(batches) >= batch_size:
                    yield batches, labels
                    batches = [] 
                    labels = []

if __name__ == '__main__':
    loader = Dataloader('cfg.json')
    cfg = loader.cfg
    env = cfg.get("env_cfg")
    model_dir = env.get("model_dir")
    # Store layers weight &amp; bias
    weights = {
    # 64 output weights = 128 (image size) / 2, needs to be 64 because max pool will halve size
    'wc1': tf.Variable(tf.random_normal([8, 8, 1, 64])), 
    # 64 input weights, 32 output weights
    'wc2': tf.Variable(tf.random_normal([8, 8, 64, 32])),
    # 32 input weights, 16 output weights 
    'wc3': tf.Variable(tf.random_normal([8, 8, 32, 16])),
    # fully connected, 16*16 image * 16 weights from output of third conv = 4096 list size.
    # I decided to have one FC neuron for every 4 inputs but you don't have to 
    'wd1': tf.Variable(tf.random_normal([16*16*16, 256])),
    'wd2': tf.Variable(tf.random_normal([256, 256])),
    'wd3': tf.Variable(tf.random_normal([256, 256])),
    'wd4': tf.Variable(tf.random_normal([256, 256])),
    'wd5': tf.Variable(tf.random_normal([256, 256])),
    'wd6': tf.Variable(tf.random_normal([256, 256])),
    'wd7': tf.Variable(tf.random_normal([256, 256])),
    'wd8': tf.Variable(tf.random_normal([256, 256])),
    'wd9': tf.Variable(tf.random_normal([256, 256])),
    'wd10': tf.Variable(tf.random_normal([256, 256])),
    # 256 outputs from FC neurons, 10 inputs to feed into softmax (class prediction)
    'out': tf.Variable(tf.random_normal([256, 10]))
    }
    biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bc3': tf.Variable(tf.random_normal([16])),
    'bd1': tf.Variable(tf.random_normal([256])),
    'bd2': tf.Variable(tf.random_normal([256])),
    'bd3': tf.Variable(tf.random_normal([256])),
    'bd4': tf.Variable(tf.random_normal([256])),
    'bd5': tf.Variable(tf.random_normal([256])),
    'bd6': tf.Variable(tf.random_normal([256])),
    'bd7': tf.Variable(tf.random_normal([256])),
    'bd8': tf.Variable(tf.random_normal([256])),
    'bd9': tf.Variable(tf.random_normal([256])),
    'bd10': tf.Variable(tf.random_normal([256])),
    'out': tf.Variable(tf.random_normal([10]))
    }
    beta = 0.001
    x = tf.placeholder(dtype = tf.float32, shape = [None, 128, 128])
    y = tf.placeholder(dtype = tf.int32, shape = [None, 10])
    logits = conv_net(x, weights, biases, 0.05)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))
    reg = L2_reg(weights['wc1']) + L2_reg(weights['wc2']) + L2_reg(weights['wd1'])
    loss = tf.reduce_mean(loss + reg * beta)
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    pred = tf.nn.softmax(logits)
    actual = tf.argmax(y, 1)
    prediction = tf.argmax(pred, 1)
    correct_pred = tf.equal(prediction, actual)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.set_random_seed(1234)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(0,24):
            X,Y = next(batch_generator(loader, env.get("train_data"), 'train', batch_size=10))
            for epoch in range(1, 20):
                _, loss_val = sess.run([train_op, loss], feed_dict={x: X, y: Y})
                print("Loss: {0}".format(loss_val))
            print("Batch: ", i)
            print("Training Accuracy: ", accuracy.eval(feed_dict={x: X, y: Y}))
        inputs_dict = {"features_data_ph": x, "labels_data_ph": y}
        outputs_dict = {"logits": logits}
        saveName = './model/runExperiments_'
        saveName = saveName + strftime("%Y-%m-%d_%H:%M:%S", gmtime())
        tf.saved_model.simple_save(sess, saveName, inputs_dict, outputs_dict)
        avg_acc = 0
        for i in range(0,12):
            X,Y = next(batch_generator(loader, env.get("valid_data"), 'valid', batch_size=50))
            acc_val = accuracy.eval(feed_dict={x: X, y: Y})
            print("Validation Accuracy: ", acc_val)
            avg_acc = avg_acc + acc_val
        avg = avg_acc/12
        print('Average Validation Accuracy: ', avg)