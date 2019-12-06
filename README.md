# fraktur
Fraktur Cracker

*** **Note: This README is not up to date**


### _getFeatures.py_
gets features for each character image. currently set to run on data in **segmentation/letters/E**.

### _zoning.py_
gets zoning features for each character image. used in **getFeatures.py**.

### _zoning_YC.py_
Yuezhen's version of zoning.py

### _char_set.txt_
list of unique chars found in data

### _no_no_chars.txt_
list of bad chars we don't want to classify. greek stuff and special symbols.

## segmentation 
folder with files for character segmentation
### _seg.py_
manually segments characters the dumb way, just using vertical lines of whitespace. 
### _seg_another.py_
segments characters with rectangular boundaries using openCV.
### _seg_the_third.py_
segments characters with flexible boundaries using openCV.
### _seg_umlaut.py_
segments characters with flexible boundaries using openCV. Also takes care of umlauts and other diacritics that aren't connected to the rest of the letter but should be part of it.
### letters
folder containing results of segmented letters from seg scripts. **man** subfolder contains results from seg.py. **E** subfolder contains select number of images of the letter 'e', for testing purposes in **zoning.py** and **getFeatures.py**.
### test_data
folder containing a select number of line images + transcriptions for testing and debugging.


## data
folder with a few books of data for testing.
