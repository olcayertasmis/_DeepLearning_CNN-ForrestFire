import keras
import numpy as np
from PIL import Image
from skimage import io

img = []
#ad = input("resim adı ")
def isfire(path,path1):
    model = keras.models.load_model(path1)
    try:
        image1 = Image.open(path)
        im2= image1.resize((250, 250))
        im2.save('x.jpg')
        image = io.imread("x.jpg")
        img.append(image)
        x = np.array(img)

        y = model.predict(x)
        print(y)
        y = np.argmax(y, axis=1)

        if y[len(y)-1] == 0:
            str1 = "0"
            return str1

        elif y[len(y)-1]:
            str1 = "1"
            return str1

    except FileNotFoundError :
        str1 ='dosya bulunamadı hata!'
        del y [0]
        return str1