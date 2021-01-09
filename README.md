This repo contains code for loading image data into a MySQL database, reading data from the database and doing transfer learning with resnet50 and logistic regression. 

All the image data belongs in "data". The data directory is then split into "test", "train" and "valid" subdirectories. 

The data is found at http://ufldl.stanford.edu/housenumbers/. The .mat files with the metadata must be extracted to .json format using the "svhn_dataextract_tojson.py" file. I decided to use the "extra" images as training data, and the "train" images as validation data. 

After setting up the directory structure and creating a MySQL db named "svhn", run "dataPopulator.py" to load the data into the database. 

To train the classifier run "runExperimentsTwo.py".

For more information on how to set up the directory structure look in the "cfg.json" file. 
