
import os, re

your_path = '/Users/Christine/cs/fraktur/data/'
os.chdir(your_path)
folders = [x for x in os.listdir() if x[0] != '.' and x[-3:] != 'txt' and x != 'letter_data']

for folder in folders:
    os.chdir(os.getcwd()+'/'+folder)
    for filename in [x for x in os.listdir() if x[0] != '.']:
        new_filename = re.sub('\.gt', '', filename)
        new_filename = re.sub('\.nrm', '', new_filename)
        new_filename = re.sub('\.bin', '', new_filename)
        os.rename(filename, new_filename)
    print('finished', folder)
    os.chdir(your_path)