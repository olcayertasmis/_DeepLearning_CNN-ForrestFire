import os                               
import time                              
import numpy as np   
import tensorflow as tf 
from PIL import Image
import numpy as np
from skimage import transform
from keras import backend as K           
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from keras.models import load_model
from keras.preprocessing import image  
from keras.utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from skimage import io
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import itertools

def train(kf,dabch,epochs,stepepoch,earlystop,test_size,rndstate,index):
    
    kfold = KFold(n_splits=kf, shuffle=True)

    test_size =  test_size / 100 
    anaKlasor = "./"

    dataSet = 'UsingDataBase/'
    trainDataSet = dataSet + 'train/'
    testDataSet = dataSet + 'test/'
    
    def readimages(kfold):
      anaKlasor2 = anaKlasor + 'dataset/'
      altKlasorler = os.listdir(anaKlasor2)


      print(altKlasorler)

      targets = ['nofire', 'fire']

      for i, split_klasor in enumerate(['train', 'test']):

          x = []
          y = []
          altKlasorler2 = os.listdir(anaKlasor2+split_klasor+"/")
          for altKlasor in altKlasorler2:
              path = anaKlasor + 'dataset/' + split_klasor + '/' + altKlasor + '/'
              print(path)

              file_names = os.listdir(path)
              for file_name in file_names:
                  image1 = Image.open(path + file_name)
                  im2= image1.resize((224, 224))
                  im2.save(path + file_name)
                  image = io.imread(path + file_name)
                  x.append(image)
                  y.append(targets.index(altKlasor))

          x = np.array(x)
          y = np.array(y)

          if i == 0:
              print(x.shape)
              X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=rndstate)
              kfoldindex=0
              # k-fold cross validation on the train and validation sets
              for train_index, val_index in kfold.split(X_train, y_train):
                  x_train_fold = np.take(X_train, train_index, axis=0)
                  x_val_fold = np.take(X_train, val_index, axis=0)

                  y_train_fold = np.take(y_train, train_index, axis=0)
                  y_val_fold = np.take(y_train, val_index, axis=0)

                  np.save("./features/x_train_fold"+str(kfoldindex)+".npy", x_train_fold)
                  np.save("./features/y_train_fold"+str(kfoldindex)+".npy", y_train_fold)
                  np.save("./features/x_val_fold"+str(kfoldindex)+".npy", x_val_fold)
                  np.save("./features/y_val_fold"+str(kfoldindex)+".npy", y_val_fold)
                  kfoldindex += 1

          elif i == 1:
              print(x.shape)
              np.save("./features/x_test.npy", X_test)
              np.save("./features/y_test.npy", y_test)
      print('train count: ' , len(y_train) , '\n')
      print('test count: ' , len(y_test) , '\n')
      print('val count: ' , len(y_val_fold) , '\n')
    
    
    
    data_aug = ImageDataGenerator(rotation_range=30, horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2)
    
    if index == 1:
        vgg_application_model = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    elif index == 2:
        vgg_application_model = tf.keras.applications.VGG19(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    else : 
        print("hata")
    #Stop trainable becuase its from keras applictions and automaticly have a default error values...
    for layer in vgg_application_model.layers:
        print(layer.name)
        layer.trainable = False

    print(len(vgg_application_model.layers))
    last_layer = vgg_application_model.get_layer('block5_pool')
    #print size of output layer....
    #print('last layer output shape:', last_layer.output_shape)
    #but it's not necessary
    last_output = last_layer.output

    x = tf.keras.layers.GlobalMaxPooling2D()(last_output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(vgg_application_model.input, x)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, decay=0.0001)


    for layer in model.layers[:15]:
        layer.trainable = False



    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    model.summary()
    #Check Point And Early Stopping
    # ModelCheckpoint callback - save best weights
    CHECKPOINT_PATH = "./checkpoint/"

    checkpoint= ModelCheckpoint(filepath=CHECKPOINT_PATH+"eyesDetectionCheckpoint.h5",
                                      save_best_only=True,
                                      verbose=1)

    # EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss',
                               patience=earlystop,
                               restore_best_weights=True,
                               mode='min')

    readimages(kfold)

    split_kategori = "test"
    x_test = np.load(anaKlasor + "/features/x_" + split_kategori + ".npy")
    y_test = np.load(anaKlasor + "/features/y_" + split_kategori + ".npy")

    

    for x in range(kf):

        split_kategori = "train_fold"+str(x)
        x_train = np.load(anaKlasor + "/features/x_" + split_kategori + ".npy")
        y_train = np.load(anaKlasor + "/features/y_" + split_kategori + ".npy")

        split_kategori = "val_fold"+str(x)
        x_val = np.load(anaKlasor + "/features/x_" + split_kategori + ".npy")
        y_val = np.load(anaKlasor + "/features/y_" + split_kategori + ".npy")



        history_model = model.fit(data_aug.flow(x_train, y_train, batch_size=dabch),
                                epochs = epochs,
                                validation_data = (x_val, y_val), 
                                verbose = 1,
                                steps_per_epoch=stepepoch,
                                validation_steps=11,
                                callbacks=[checkpoint , early_stop])
        model_json = model.to_json()

        model.save(CHECKPOINT_PATH+"eyesDetectionCheckpoint.h5")
        with open("./checkpoint/eyesDetectionJson{}.json".format(x), "w") as json_file:
            json_file.write(model_json)
        # here for saving model history in dectinory library...
        history_file = './checkpoint/eyesDetectionHistory{}.npy'.format(x)
        np.save(history_file ,history_model.history)

    model.load_weights(CHECKPOINT_PATH+"eyesDetectionCheckpoint.h5")

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
        
        

    y_preds = model.predict(x_test)
    print(y_preds)
    y_preds = np.argmax(y_preds, axis=1)
    
    loaded_history = np.load(history_file , allow_pickle='TRUE').item()
    
    print(y_preds)
    print(y_test)
    cm = confusion_matrix(y_test,y_preds)
    basari = accuracy_score(y_test, y_preds) * 100
    print("Accuracy:", basari)
    func_model_ciz(cm)

    print(cm)
    
