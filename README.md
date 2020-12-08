# Convolutional Neural Networks (CNN) for Image Classification
### CIFAR-10 Image Classification

This project uses images and labels from the CIFAR-10 dataset to build an image classification model.  The data can be loaded directly using the tensorflow package using the code below.

```
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
```

Our goal is to correctly predict image class (from 10 classes) based on the pixels that make up each image.  Specifically, we aim to build a model that maximizes overall accuracy.

**Problem Description:**

The dataset that we use to build and test our model contains 60,000 color images, each shaped as a 32x32 pixel square. Our target has 10 equally balanced classes.

Technically, we could try to build a fully connected neural network to make predictions, but the large amount of data would be quite computationally expensive - 32x32 pixels * 3 RGB = 3072 values associated with each image (and that's just for a tiny 32x32 image).  To address this, we will build a convolutional neural network to window through our sections of each image.
