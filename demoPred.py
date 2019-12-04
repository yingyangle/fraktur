import pickle
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from getTestFeatures # import make a small getfeatures function for testing
your_path_here = '/Users/ovoowo/Desktop/fraktur/'
#your_path_here = '/Users/Christine/cs/fraktur/'
datapath = your_path_here+'data/dataset/'

def randomPrediction(path,model):
    files = os.listdir(path)
    n = len(files)
    testID = np.random.randint(n-1,size = 10)
    testID[0]
    presentFile = [files[id] for id in testID]
    for file in presentFile:
        img = mpimg.imread(datapath+file) #Importing image data into Numpy arrays
        imgplot = plt.imshow(img)
        (black,dist,label) = getFeats(file,8)
        pred = chr(int(model.predict(black.reshape(1, -1))))
        # pred = model.predict(dist)
        charLabel = chr(int(label))
        plt.title('Label: '+pred+' Prediction: '+pred)
        plt.show()
