# How to organize your Machine Learning workflow

In this article, we are going to see how you should organize your machine learning workflow. I also include some examples using python's libraries.

Here are the different step we are going to go through.

- Introduction

I. Data loading and overview
- Loading the data
- Overview

II. Data cleaning
- Duplicated and missing values
- Deal with outliers

III. Features engineering  
- Transformations
- Features creations and deletions
- Dimensional reductions

IV. Model selection  
- Split
- Metrics
- Models stability (Cross-Validation)
- Check over/underfit

V. Hyperparameters tuning  

VI. Training and predictions  


## 0. Introduction and requirements

Before starting, you need to setup your developing environement. If you didn’t, please follow this easy tutorial to get started.
Also, be aware that some bullets points highlighted below imply a basic understanding of different mathematic concepts. I highly recommand you read/keep aside this article on Statistics Basics if you are not confident at all with mathematics.


## I. Data loading and overview

The very first step is to import the essential libraries.

```python
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

%matplotlib inline
```

Now we load the data.

```python
FILEPATH = os.path.join('data', 'dataset.csv')

df = pd.read_csv(FILEPATH, index_col=0)
df.head(2)
```

### b. Overview

In Machine Learning, we want to build a model capable of predicting one of these variables (called the target) thanks to the others (called the features). Here, our target will be variable3 and our features to do it variable1 and 2. We say that our model has currently two dimensions (i.e. two features).

First of all, we want to know what kind of values we want to predict. This will tell us what kind of algorithms use to build our model. The variables can be:
categorical (qualitative) : who/what/what kind
or numerical (quantitative) : how much

If the target is categorical, we will make a classification. If the target is numerical, we will make a regression.
If we don’t have the target values or/and don’t know what we are really looking for, we will use clustering algorithms.

Note: When the target is provided (labeled data), the learning is supervised (classications and regressions). When it’s not, the learning is unsupervised (clusterings).

Now we have a purpose and a way to get to that. Time to clean our data before starting to think our model.
