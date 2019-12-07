import os,numpy as np
your_path_here = '/Users/ovoowo/Desktop/'
#your_path_here = '/Users/Christine/cs/'
modulepath = your_path_here +'fraktur/features/'
os.chdir(modulepath)
from getFeatures import getFeats, createDataset
storepath = your_path_here +'fraktur/cleaning/' #the place to store imgs/error message
booksfolder = 'testdata/letter_data/' #the one store books'imgs we want to clean
imgpath =your_path_here +'fraktur/'+booksfolder
txtpath = storepath + 'letterfeatures/'

# # # # # # # # test # # # # # # # #
# test = '/Users/ovoowo/Desktop/fraktur/data/letter_data/109m'
# os.chdir(test)
# folder = '109m'
# txtGenerator(your_path_here+'fraktur/',test,0,folder,6)
ignore_folders = ['0','1797-wackenroder_herzensergiessungen',
        '1802-novalis_ofterdingen','1804-paul_flegeljahre01','1815-hoffmann_elixiere01',
        '1816-perthes_buchhandel','1817-hoffmann_nachtstuecke01','1819-goerres_revolution',
        '1821-mueller_waldhornist','1826-eichendorff_taugenichts','1827-clauren_liebe'
        ,'1827-heine_lieder','1827-heine_reisebilder02']
'''
######################
clean 'fraktur/cleaning/letterfeatures/' before run
########################
'''
#get all the folder's name that need to be cleaned
os.chdir(imgpath)
books = os.listdir(imgpath)[1:] #to create folder for each book / or not
for f in ignore_folders: # remove folders we don't wanna check
    try: books.remove(f)
    except: print('failed to remove', f)
tracker = 0
for book in books: #for each book folder
    bookpath = imgpath+book+'/'
    letterFolders= os.listdir(bookpath)[1:] #get rid of .DS_Store document each letters
    count = 0
    total = len(letterFolders)
    os.chdir(txtpath)
    os.mkdir(book)
    lstorepath =txtpath+book+'/'
    for folder in letterFolders: #access the letter folder for one book
        #os.chdir(imgpath+folder) (storepath,datapath,mode,foldername,n)
        letterpath = bookpath +folder+'/'
        # get feature for every letter folder
        createDataset(lstorepath, letterpath, 6, 0, 0) # black feats
        createDataset(lstorepath, letterpath, 6, 0, 1) # dist feats
        count += 1
        print('~'*40)
        print('Folder '+folder+' is done. \t '+str(count)+' out of '+str(total)+' folder left')
        print('~'*40)
    ###Finish one book, move to next book
    ######test######
    if book == '0':
        break
    tracker += 1
    print('='*50)
    print(str(tracker)+' out of '+str(len(books))+' book finished')
    print('='*50)
