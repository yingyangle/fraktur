import pickle,os, numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from getTestFeatures import getFeats, getData # import make a small getfeatures function for testing
your_path_here = '/Users/ovoowo/Desktop/fraktur/'
#your_path_here = '/Users/Christine/cs/fraktur/'
demopath = your_path_here+'demo/'

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

os.chdir(demopath)
model = pickle.load(open('5_8_NN.sav','rb'))
#########################################################
#######Run it once for featuretxt then comment out#######
#imgs = [x for x in os.listdir() if x[-3:] == 'png']
#for img in imgs:
#    Bdataset = []
#    (black,label) = getFeats(demopath,img,8)
#    Bdata = np.concatenate((black,label))
#    Bdataset.append(Bdata)
#    Btestdata = np.array(Bdataset)
#    np.savetxt(img+'.txt',Btestdata, delimiter=', ', fmt='%12.8f')
#########################################################
for txt in [x for x in os.listdir() if x[-3:] == 'txt']:
    (features,label) = getData(txt,demopath)
    demoPrediction(features,label,model,demopath+txt[:-4])

