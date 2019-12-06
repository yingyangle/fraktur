import os
from os.path import join

datapath = '/Users/Christine/Downloads/GT4HistOCR/corpus/RIDGES-Fraktur'
datapath = '/Users/Christine/Downloads/GT4HistOCR/corpus/dta19'
# list of book folders
folders = [x for x in os.listdir(datapath) if os.path.isdir(join(datapath, x))]
folders.sort()
names = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel', \
'india', 'juliett', 'kilo', 'lima', 'mike', 'november', 'oscar', 'papa', 'quebec', \
'romeo', 'sierra', 'tango', 'uniform', 'victor', 'whiskey', 'xray', 'yankee', 'zulu'\
'anton', 'ärger', 'berta', 'cäsar', 'charlotte', 'dora', 'emil', 'friedrich', 'gustav'\
'heinrich', 'ida', 'julius', 'kaufmann', 'ludwig', 'martha', 'nordpol', 'otto', 'österreich']

directory_name = 'directory_'+datapath[datapath.rfind('/')+1:]+'.txt'
aus = open(directory_name, 'a')

for i in range(len(folders)):
    folder = folders[i]
    year = folder[:4]
    aus.write(names[i]+' - '+folder+'\n')
    os.rename(join(datapath,folder), join(datapath,year+names[i]))

aus.close()
