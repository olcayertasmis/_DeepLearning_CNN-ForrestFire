import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
from sklearn.metrics import roc_auc_score, roc_curve
import os
from skimage import io
import keras
import numpy as np
import skimage
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
import zipfile
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
import random
from skimage import filters
from keras.utils import to_categorical
import os
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense


def isCheck_Klasor(path):
    if not os.path.isdir(path):
        os.mkdir(path, 755);


def perf_measure(cm, alg_name, etiket):
    print("Confusion Matrix:")
    print(cm)
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    acc = round(float(TP + TN) / (TP + TN + FP + FN) * 100, 2)
    sensitivity = round(float(TP) / (TP + FN) * 100, 2)
    specificity = round(float(TN) / (FP + TN) * 100, 2)
    precision = round(float(TP) / (TP + FP) * 100, 2)
    F1_score = round(2 / ((1 / sensitivity) + (1 / precision)), 2)
    print(alg_name, " Algoritması ", etiket, " Dataset, Acc:", acc, "Sen:", sensitivity, " Spe:", specificity, " Pre:",
          precision, " F1-score:", F1_score)
    # basarilar.append([alg_name,etiket,acc,sensitivity,specificity,precision,F1_score])


def cm_analysis(y_true, y_pred, etiket, labels, alg_name, figsize2=(5, 5)):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    perf_measure(cm, alg_name, etiket)

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)

    labels2 = labels

    cm = pd.DataFrame(cm, index=labels2, columns=labels2)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize2)
    sns.heatmap(cm, cmap="OrRd", annot=annot, fmt='', ax=ax)
    plt.title('Confusion matrix of the ' + alg_name)
    plt.show()
    plt.savefig()


model_name = "MyCNN"
anaKlasor = "./"
model = keras.models.load_model('./models/model_myCNN.h5')

split_kategori = "test"
x_test = np.load(anaKlasor + "/features/x_" + split_kategori + ".npy")
y_test = np.load(anaKlasor + "/features/y_" + split_kategori + ".npy")

y_preds = model.predict(x_test)
print(y_preds)
y_preds = np.argmax(y_preds, axis=1)
print(y_preds)
print(y_test)

labels = [0, 1]
cm_analysis(y_test, y_preds, "dataset", labels, "by " + model_name, figsize2=(5, 5))

imgplot = plt.imshow(x_test[0])
plt.show()

image = np.array(x_test[len(y_test) - 1])
print(image.shape)
result = model.predict(np.expand_dims(image, axis=0))
print(result)
result = np.argmax(result, axis=1)
print(result, y_test[len(y_test) - 1])