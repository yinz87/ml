# -*- coding: utf-8 -*-
"""
Created on Tue May  8 11:01:09 2018

@author: 100314426
"""

import math
from IPython import display
from matplotlib import cm, gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

datasets = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv",sep=",")

datasets = datasets.reindex(np.random.permutation(datasets.index))

datasets["median_house_value"]/= 1000.0

my_feature = datasets[["total_rooms"]]


feature_columns = [tf.feature_column.numeric_column("total_rooms")]

targets = datasets["median_house_value"]

my_optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.11)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)


linear_regressor = tf.estimator.LinearRegressor(feature_columns = feature_columns, optimizer = my_optimizer)



def my_input_fn(features,targets,batch_size = 1, shuffle = True, num_epochs = None):
    features = {key:np.array(value) for key,value in dict(features).items()}
    
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    if shuffle:
        ds = ds.shuffle(buffer_size = 10000)
        
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


_ = linear_regressor.train(input_fn = lambda:my_input_fn(my_feature,targets)
,steps=100)

prediction_input_fn = lambda:my_input_fn(my_feature,targets,num_epochs= 1, shuffle = False)

predictions = linear_regressor.predict(input_fn=prediction_input_fn)
predictions = np.array([item['predictions'][0] for item in predictions])

mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)

min_house_value = datasets["median_house_value"].min()

max_house_value = datasets["median_house_value"].max()

min_max_differ = max_house_value - min_house_value

print ("min. value: %0.3f" %min_house_value)
print ("max. value: %0.3f" %max_house_value)
print ("min_max_differ: %0.3f" %min_max_differ)
print ("root mean square: %0.3f" %root_mean_squared_error)

compare = pd.DataFrame()
compare["predictions"] = pd.Series(predictions)
compare["targets"] = pd.Series(targets)
print (compare.describe())

sample = datasets.sample(n=300)

x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')[0]

y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias

plt.plot([x_0,x_1],[y_0,y_1],c = "r")

plt.scatter(sample["total_rooms"], sample["median_house_value"])
plt.show()