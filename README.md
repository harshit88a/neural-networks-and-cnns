# Neural Networks, CNNs, and LSTMs

This is a part of my assignment from my CSE 676 - Deep Learning course. This is divided into four parts:

## Part 1: CNN using EMNIST

File : cnn_emnist.ipynb

### Dataset Used:

EMNIST (https://arxiv.org/pdf/1702.05373)

Dataset consists of total 100800 samples and 36 classes (26 alphabets and 10 numerals)

### CNN Network Architecture

This CNN model consist of 5 hidden layers

Conv2d (layer 1) -> Maxpool -> Conv2d (layer 2) -> Maxpool ->Conv2d (layer 3) -> Maxpool -> Linear (layer 4) -> Dropout -> Linear (layer 5)

Total params: 134,244

Trainable params: 134,244

Non-trainable params: 0

Total mult-adds (Units.MEGABYTES): 368.90

## Part 2: VGG-16 and Resnet-18 Implementation for image classification

File: vgg16_and_resnet18.ipynb

Dataset used: Dogs, Cars, and food (10000 images per class)

This project implements and compares one of the fundamentals CNN architectures: VGG-16 (Version C) and ResNet-18 for image classification. Advanced techniques were explored to improve model performance and the transition from standard deep CNNs to networks with residual connections.


## Part 3: Implement Resnet-34 using transfer learning

File: resnet34.ipynb

Dataset: Flower Dataset (https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

This project implements both Resnet-34 using pretrained weights on flower dataset to classify a flower into given 102 labels.


## Part 4: Sentiment analysis using LSTM

File: sentiment_analysis_using_LSTM.ipynb

Dataset used: Twitter US Airline Sentiment (https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

This part implements LSTM and GRU networks on US airline tweet dataset. This dataset is of top US Airline reviews from Twitter. This dataset was scraped from February 2015 and classified into 3 sentiments: Positive, Negative, and Neutral tweets. Some basic NLP techniques were also used to extract useful features from the dataset.

