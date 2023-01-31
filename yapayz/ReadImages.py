import os
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split

anaKlasor = "./CNN-Forrestfire-binary-classification-master/"

anaKlasor2 = anaKlasor + 'dataset/'
altKlasorler = os.listdir("C:/Users/Monster/Desktop/CNN-Forrestfire-binary-classification-master/CNN-Forrestfire-binary-classification-master/dataset")


print(altKlasorler)

targets = ['nofire', 'fire']

for i, split_klasor in enumerate(['Training-validation', 'Testing']):

    x = []
    y = []
    altKlasorler2 = os.listdir(anaKlasor2+split_klasor+"/")
    for altKlasor in altKlasorler2:
        path = anaKlasor + 'dataset/' + split_klasor + '/' + altKlasor + '/'
        print(path)

        file_names = os.listdir(path)
        for file_name in file_names:
            # print (file_name,">>>>",targets.index(altKlasor))
            image = io.imread(path + file_name)
            x.append(image)
            y.append(targets.index(altKlasor))

    x = np.array(x)

    if i == 0:
        print(x.shape)
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=42)

        np.save(anaKlasor+"features/x_train.npy", X_train)
        np.save(anaKlasor+"features/y_train.npy", y_train)

        np.save(anaKlasor+"features/x_val.npy", X_val)
        np.save(anaKlasor+"features/y_val.npy", y_val)

    elif i == 1:
        print(x.shape)
        np.save(anaKlasor+"features/x_test.npy", x)
        np.save(anaKlasor+"features/y_test.npy", y)