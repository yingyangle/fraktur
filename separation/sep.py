import os, re
from PIL import Image
import more_itertools as mit

ein = open('hard2.txt')
txt = ein.read()[:-1]
ein.close()

im = Image.open('b.png', 'r') # open image
width, height = im.size # image size
pix_val = list(im.getdata()) # pixel color values

cols = [pix_val[n::width] for n in range(width)] # pixels as columns
print('\n'.join(' '.join(map(str,sl)) for sl in cols)) # print cols

# get column indices of whitespace
whitespace = []
for i in range(len(cols)):
    c = cols[i]
    whites = [x for x in c if x >= 200] # whitespace pixels
    if len(whites) + 3 >= len(c): whitespace.append(i) # allow 3 non-whitespace
    # all(x >= 200 for x in c)

# group consecutive whitespace
groups = [list(group) for group in mit.consecutive_groups(whitespace)]

try: # make folder for cropped letter images
    os.mkdir('letters')
    os.chdir('letters')
except: os.chdir('letters')

for i in range(len(groups)-1): # crop letter images
    left = groups[i][-1]
    right = groups[i+1][0]
    area = (left, 0, right, height)
    cropped_im = im.crop(area)
    # cropped_im.show()
    cropped_im.save(str(i)+'.png')
