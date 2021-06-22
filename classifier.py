import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X=np.load('image.npz')["arr_0"]
y=pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses=len(classes)

xTrain,xTest,yTrain,yTest=train_test_split(X,y,random_state=9,train_size=3500,test_size=500)
xTrainScaled=xTrain/255.0
yTrainScaled=xTest/255.0
clf=LogisticRegression(solver="saga",multi_class="multinomial").fit(xTrainScaled,yTrain)

def getPrediction(image):
    imagePillow=Image.open(image)
    imageBw=imagePillow.convert("L")
    imageResize=imageBw.resize((28,28),Image.ANTIALIAS)
    pixelFilter=20
    minPixel=np.percentile(imageResize,pixelFilter)
    imageScaled=np.clip(imageResize-minPixel,0,255)
    maxPixel=np.max(imageResize)
    imageScaled=np.asarray(imageScaled)/maxPixel
    testSample=np.array(imageScaled).reshape(1,784)
    prediction=clf.predict(testSample)
    return prediction[0]