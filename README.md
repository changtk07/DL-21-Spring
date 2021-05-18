# DL-21-Spring
Homeworks and final project for Deep Learning, NYU 2021 Spring.

You can find class repo here: https://github.com/Atcold/NYU-DLSP21


## Final Project

This final project is a image classification task with 25,600 labeld train images, 25,600 labeled validation images and 512,000 unlabeled images. There are a total of 800 classes.

We used MoCo as our semi-supervised learning framework and tried to combine design choices from other models. Most of the codes is from https://github.com/facebookresearch/moco.

We reached 30.48% accuracy with 250 epchos of pretrain + 100 epochs frozen-features classifier + 100 epochs unfrozen-features classfier.
