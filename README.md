This repo contains code for loading image data into a MySQL database, reading data from the database and running a CNN classifier on the data. 

All code should be in a directory called "code", whereas all the image data belongs in "data". The data directory is then split into "test", "train" and "valid" subdirectories. The .json metadata files stay in the top level directory, but the image files must be placed in the appropriate subdirectories. 

The data is found at http://ufldl.stanford.edu/housenumbers/. The .mat files with the metadata must be extracted to .json format using the "svhn_dataextract_tojson.py" file. I decided to use the "extra" images as training data, and the "train" images as validation data. 

After setting up the directory structure and creating a MySQL db named "svhn", run "dataPopulator.py" to load the data into the database. 

To train the classifier run "runExperiments.py".

For more information on how to set up the directory structure look in the "cfg.json" file. 


For some resources on CNN's:

http://www.jessicayung.com/explaining-tensorflow-code-for-a-convolutional-neural-network/

http://cs231n.github.io/convolutional-networks/#layers

https://arxiv.org/pdf/1603.07285v1.pdf


'SAME' padding and the intuition behind the architecture and weight shapes:

https://stackoverflow.com/questions/48491728/what-is-the-behavior-of-same-padding-when-stride-is-greater-than-1

https://www.tensorflow.org/api_guides/python/nn#Convolution

https://stackoverflow.com/questions/46465925/input-to-reshape-is-a-tensor-with-37632-values-but-the-requested-shape-has-1505/52240509#52240509