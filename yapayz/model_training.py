import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import accuracy_score

import matplotlib
from sklearn.metrics import roc_auc_score, roc_curve
import os
from skimage import io
import keras
import itertools
import numpy as np
import skimage
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
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
        os.mkdir(path, 755)


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
    # plt.savefig(filename)
    plt.title('Confusion matrix of the ' + alg_name)
    plt.show()


def func_model_kaydet_ve_ciz(model_bilgisi, model_log, y_preds, y_val_preds, egitimBilgi, label):
    anaKlasor = "./yapayz/"
    # plotting the metrics
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(model_log.history['accuracy'])
    plt.plot(model_log.history['val_accuracy'])
    plt.title('model accuracy with ' + model_bilgisi)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'validation accuracy'], loc='lower right')
    plt.savefig(anaKlasor + "/acc_loss_plots/" + model_bilgisi + "_acc.png")

    plt.subplot(2, 1, 2)
    plt.plot(model_log.history['loss'])
    plt.plot(model_log.history['val_loss'])
    plt.title('model loss with ' + model_bilgisi)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'validation loss'], loc='upper right')
    plt.tight_layout()
    plt.savefig(anaKlasor + "/acc_loss_plots/" + model_bilgisi + "_loss.png")

    np.save(anaKlasor + "/sonuclar/y_preds_by_" + model_bilgisi + ".npy", y_preds)
    np.save(anaKlasor + "/sonuclar/y_val_preds_by_" + model_bilgisi + ".npy", y_val_preds)
    np.save(anaKlasor + "/sonuclar/model_log_by_" + model_bilgisi + ".npy", model_log.history)

    np.save(anaKlasor + "/sonuclar/" + label + "_egitimBilgi.npy", egitimBilgi)

    print(model_bilgisi, " kayıt bitti...")


def func_model_kaydet_ve_ciz2(model_bilgisi, model_log):
    anaKlasor = "./yapayz"
    # plotting the metrics
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(model_log.history['accuracy'])
    plt.plot(model_log.history['val_accuracy'])
    plt.title('model accuracy with ' + model_bilgisi)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'validation accuracy'], loc='lower right')
    plt.savefig(anaKlasor + "/acc_loss_plots/" + model_bilgisi + "_acc.png")

    plt.subplot(2, 1, 2)
    plt.plot(model_log.history['loss'])
    plt.plot(model_log.history['val_loss'])
    plt.title('model loss with ' + model_bilgisi)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'validation loss'], loc='upper right')
    plt.tight_layout()
    plt.savefig(anaKlasor + "/acc_loss_plots/" + model_bilgisi + "_loss.png")

    print(model_bilgisi, " kayıt bitti...")


def get_model(num_classes):
    model = Sequential()
    model.add(Conv2D(8, (3, 3), padding="same", input_shape=(250, 250, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.summary()
    return model
def func_model_ciz(model_bilgisi):
        # plotting the metrics

        plt.imshow(model_bilgisi, cmap='Blues',interpolation='nearest')
        plt.xticks([0, 1], ['True', 'False'])
        plt.yticks([0, 1], ['True', 'False'])
        for i, j in itertools.product(range(model_bilgisi.shape[0]), range(model_bilgisi.shape[1])):
            plt.text(j, i, model_bilgisi[i, j], horizontalalignment='center', color='black')
        plt.colorbar()
        #plt.show()
        plt.savefig("./_confusion_matrix.png")

def main(epoch,dabch):
    anaKlasor = "./yapayz/"
    isCheck_Klasor(anaKlasor + "features")
    split_kategori = "train"
    x_train = np.load(anaKlasor + "features/x_" + split_kategori + ".npy")
    y_train = np.load(anaKlasor + "features/y_" + split_kategori + ".npy")

    split_kategori = "val"
    x_val = np.load(anaKlasor + "features/x_" + split_kategori + ".npy")
    y_val = np.load(anaKlasor + "features/y_" + split_kategori + ".npy")

    split_kategori = "test"
    x_test = np.load(anaKlasor + "features/x_" + split_kategori + ".npy")
    y_test = np.load(anaKlasor + "features/y_" + split_kategori + ".npy")

    isCheck_Klasor(anaKlasor + "acc_loss_plots/")
    isCheck_Klasor(anaKlasor + "sonuclar/")

    num_classes = 2
    model = get_model(num_classes)

    label = "MyCNN"

    y_train_categorical = to_categorical(y_train)
    y_val_categorical = to_categorical(y_val)
    y_test_categorical = to_categorical(y_test)

    print(y_train)
    print(y_train_categorical)

    print(y_train_categorical.shape, y_val_categorical.shape, x_train.shape, x_val.shape)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    epoch_value = epoch
    model_log = model.fit(x_train, y_train_categorical, validation_data=(x_val, y_val_categorical), verbose=1,
                          epochs=epoch_value, batch_size=dabch)
    print(label, "Model egitimi bitti...")

    isCheck_Klasor('./models/')
    model.save('./models/model_myCNN.h5')

    y_val_preds = model.predict(x_val)
    y_val_preds = np.argmax(y_val_preds, axis=1)

    y_preds = model.predict(x_test)
    y_preds = np.argmax(y_preds, axis=1)
    cm = confusion_matrix(y_test,y_preds)
    basari = accuracy_score(y_test, y_preds) * 100
    print("Accuracy:", basari)
    func_model_ciz(cm)
    bilgi = [label + "->data org->" + str(epoch_value) + "epoch ile çalışıldı..."]


    func_model_kaydet_ve_ciz(label,model_log,y_preds,y_val_preds,bilgi,label)

    #func_model_kaydet_ve_ciz(label, model_log)