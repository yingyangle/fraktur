import pickle,os, numpy as np
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.neighbors import KNeighborsClassifier
from getTestFeatures import getFeats, getData 
# import make a small getfeatures function for testing

#your_path_here = '/Users/ovoowo/Desktop/fraktur/'
your_path_here = '/Users/Christine/cs/fraktur/'

def demoPrediction(features,label,model,filepath):
    print(filepath)
    # files = os.listdir(path)
    # n = len(files)
    # testID = np.random.randint(n-1,size = 10)
    # testID[0]
    # presentFile = [files[id] for id in testID]
    # for file in presentFile:
    img = mpimg.imread(filepath) #Importing image data into Numpy arrays
    imgplot = plt.imshow(img)
    pred = chr(int(model.predict(features.reshape(1, -1))))
    charLabel = chr(int(label))
    plt.title('Label: '+charLabel+' Prediction: '+pred)
    plt.show()
    return


# execute 

demopath = your_path_here+'demo/'
os.chdir(demopath)
model = pickle.load(open('5_8_NN.sav','rb')) # load trained model

#########################################################
####### Run this once for featuretxt then comment it out #######
#imgs = [x for x in os.listdir() if x[-3:] == 'png']
#for img in imgs:
#    Bdataset = []
#    (black, label) = getFeats(demopath,img,8)
#    Bdata = np.concatenate((black,label))
#    Bdataset.append(Bdata)
#    Btestdata = np.array(Bdataset)
#    np.savetxt(img+'.txt', Btestdata, delimiter=', ', fmt='%12.8f')
#########################################################

# show model prediction and actual label for each img
for txt in [x for x in os.listdir() if x[-3:] == 'txt']:
    (features, label) = getData(txt, demopath)
    demoPrediction(features, label, model, demopath+txt[:-4])

