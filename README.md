# Fast building damage assessment using satellite imagery #

### What is this repository for? ###

We use a convolutional neural network to detect buildings in satellite data. We have tried several models
and acheive >0.6 intersection over union. We use our trained model on images of areas affected by natural or
 manmade disasters to estimate the number of buildings distroyed.

A result example from our neural net model can be seen below
<!-- [](https://github.com/amirdel/stanfordHacks/blob/master/presentation/5million_param_model/test_thresh.png) -->
![threshold image](/presentation/5million_param_model/test_thresh.png?raw=true "Optional Title")

The number of buildings in these threshold images are counted using by detecting the contected blobs in the
images
![threshold image](/presentation/counting_buildings.JPG?raw=true "Optional Title")


<!-- ### How do I set it up? ### -->

<!-- * First clone this repository. -->
<!-- * run this command to create the conda env: `conda env create -f planet_pipeline/env.yml` -->
