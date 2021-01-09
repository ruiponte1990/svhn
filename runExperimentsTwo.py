import os
import statistics
import numpy as np
import pickle
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from dataloader import Dataloader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix



def extract_features(X,model, batch_size):
    batchImages = np.vstack(X)
    features = model.predict(batchImages,batch_size=batch_size)
    features = features.reshape((features.shape[0],7*7*2048))
    return features

def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    process_image = preprocess_input(resized_image)
    img_expand = np.expand_dims(process_image, axis=0)
    # label = tf.convert_to_tensor(label.reshape(1,10)) # use for NN after transfer, labels as 100 size array of one hot encoding arrays
    label = np.argmax(label) + 1 # use for log regression after transfer, labels as 1-D array of numbers
    return img_expand, label

def batch_generator(model, loader, file_dir, table_name, batch_size=10):
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
                    X = [item[0] for item in boxes][0]
                    Y = [item[1][0] for item in boxes][0]
                    X, Y = preprocess(X, Y)
                    batches.append(X)
                    labels.append(Y)
                except Exception:
                    loader.log.error('Error getting data from file: ', f)
                    continue
                if len(batches) >= batch_size:
                    batches = extract_features(batches, model, batch_size)
                    # labels = np.vstack(labels) # need to vstack features for NN
                    yield batches, labels
                    batches = [] 
                    labels = []


if __name__ == '__main__':
    loader = Dataloader('cfg.json')
    cfg = loader.cfg
    env = cfg.get("env_cfg")
    model = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=3000)
    conv_base = ResNet50(weights='imagenet', include_top=False)
    for i in range(0,2):
        train_x, train_y = next(batch_generator(conv_base, loader, env.get("train_data"), 'train', batch_size=1000))    
        model.fit(train_x, train_y)
    valid_x, valid_y = next(batch_generator(conv_base, loader, env.get("valid_data"), 'valid', batch_size=1000))
    preds = model.predict(valid_x)
    print(classification_report(valid_y, preds))

    # plt.figure(figsize=(12, 8))
    # cm = confusion_matrix(y_true=train_y, y_pred=preds)
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0
    # sns.heatmap(cm, annot=True, cmap='Reds', fmt='.1f', square=True)
    
    print("[INFO] saving model...")
    f = open("models/logReg.cpickle", "wb")
    f.write(pickle.dumps(model))
    f.close()
