import os, numpy as np, pickle

mypath = '/Users/ovoowo/Desktop/fraktur/testdata/letter_data/'
books = os.listdir(mypath)[1:]
books.remove('errors.txt')
books.remove('words')
#lens = [np.array(os.listdir(mypath+book)).size for book in books]
#ind = np.argmax(np.array(lens))
#prototype = books[ind]
#protopath = mypath+prototype+'/'

chars = []
for book in books: #every books
    protopath = mypath+book+'/'#every books's letter folders
    folders = os.listdir(protopath)[1:]
    labels = []
    for folder in folders:
        id = 0
        while id!= len(folder):
            try:
                n =folder[id]
                label = int(n)
            except:
                labels.append(folder[id:])
            id += 1
    newlabels = [ x for x in labels if len(x)!=1 ] # truncated label for one book

    for label in newlabels:
        temp = label
        id = -1
        while id != -len(label):
            try:
                int(label[id]) #if label[i] get the number eg 100d
                break
            except:
                temp = label[id:] #label
            id -= 1
        if temp not in chars:
            chars.append(temp)
    keys = [x for x in chars if len(x)!=1]

dict ={} #Get the frequency
value = -1
for key in keys:
    keys = dict.keys()
    if key not in keys:
        dict[key] = value
        value -= 1
print(dict)
name = 'dictionary.sav'
featurepath = '/Users/ovoowo/Desktop/fraktur/features/'
pickle.dump(dict, open(featurepath+name,'wb'))
