import os, shutil, numpy as np
from wordSeg import getLabels

def letterCounter():
    your_path_here = '/Users/ovoowo/Desktop/fraktur'
    # your_path_here = '/Users/Christine/cs/fraktur'
    path = your_path_here + '/segmentation'
    # list of book folders in 'data'
    os.chdir(your_path_here + '/data')
    folders = [x for x in os.listdir() if os.path.isdir(os.path.join(os.getcwd(), x))]
    folders.remove('letter_data')
    folders.remove('dataset')

    letters = []
    for foldername in folders: # for each book folder in 'data'
        datapath = your_path_here+'/data/'+foldername # path to img/txt data for this book
        os.chdir(datapath)
        txts = [x for x in os.listdir() if x[-3:] == 'txt']
        for txt in txts:
            w = getLabels(txt)
            word_ls = [ w[i:i+1] for i in range(len(w))]
            letters.append(np.array(word_ls))
    l_list = np.concatenate(np.array(letters)).tolist()
    print(type(l_list[0]),len(l_list))
    print(l_list[0],l_list[1])
    freq={}
    for l in l_list:
        keys = freq.keys()
        if l in keys:
            freq[l] += 1
        else:
            freq[l] = 1
    print('Total number of chars = ',len(freq.keys()))
    print ("Char - frequency : \n")
    keyList = list(freq.keys())
    # for i in range(len(keyList)):
    #     print('The character '+keyList[i]+' has frequency: '+str(freq[keyList[i]]))
    temp = [print('The character '+keyList[i]+' has frequency: '+str(freq[keyList[i]])) for i in range(len(keyList))]
    return freq
