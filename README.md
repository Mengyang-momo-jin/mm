# DaD-Framework

Download the data from [this](https://vision.cs.uiuc.edu/attributes/) link and place in apascal/VOC2008 folder.

###### Training

Run bin/train_attributes.py to start training. The model will be saved in bin/snapshots.

###### Inference

Download this [model file](https://drive.google.com/file/d/1YCrEADP4OcATguZr3CtNHdAdoEgks4Uu/view?usp=sharing) and place in bin/snapshots folder.

Run bin/evaluate_attributes.py to evaluate the model on test data.

This repository is an extension of [this](https://github.com/fizyr/keras-retinanet) repository, check for detailed instructions.
