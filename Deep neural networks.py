import tensorflow.compat.v1 as tf
import tensorflow as tf1
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import csv
plt.ion()
plt.show()

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# normalization
Normalization_data = pd.read_csv('Normalization.CSV',encoding='gbk')
Normalization_Input=Normalization_data[['position','frequency']]
Normalization = MinMaxScaler(feature_range=(0,1))
Normalization_output=Normalization.fit_transform(Normalization_Input)
# train data
Training_Data = pd.read_csv('Train data.CSV',encoding='gbk')
Input_data=Training_Data[['position','frequency']]
Input_data_Normalization=Normalization.transform(Input_data)
Input_data=np.array(Input_data_Normalization,dtype='float32')
Output_data=Training_Data[['ave-field']]
Output_data=np.array(Output_data,dtype='float32')

# test data
test_data = pd.read_csv('Test data.CSV',encoding='gbk')
position=test_data[['position']]
position=np.array(position,dtype='float32')
test_input=test_data[['position','frequency']]
test_input_Normalization=Normalization.transform(test_input)
test_input_Normalization=np.array(test_input_Normalization,dtype='float32')

# input
Net_Input = tf.placeholder(tf.float32, [None,2])
Net_Output = tf.placeholder(tf.float32, [None,1])

#hidden layer
l1 = add_layer(Net_Input, 2, 30, activation_function=tf1.nn.relu)
l2 = add_layer(l1, 30, 30, activation_function=tf1.tanh)
l3 = add_layer(l2, 30, 30, activation_function=tf1.tanh)
l4 = add_layer(l3, 30, 30, activation_function=tf1.sigmoid)

#output
prediction = add_layer(l4, 30, 1, activation_function=None)
loss = tf.reduce_mean(tf.square(Net_Output - prediction))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
     sess.run(init)
     is_train = False
     saver = tf.train.Saver(max_to_keep=1)

     if is_train:
      model_file = tf1.train.latest_checkpoint('save/')
      saver.restore(sess, model_file)
      for i in range(10001):
       sess.run(train_step, feed_dict={Net_Input: Input_data, Net_Output: Output_data})
       if i % 100 == 0:
         print(sess.run(loss, feed_dict={Net_Input: Input_data, Net_Output: Output_data}))
      saver.save(sess, 'save/model', global_step=i + 1)

     else:
         model_file = tf1.train.latest_checkpoint('save/')
         saver.restore(sess, model_file)
         with open("Test-pre.csv", "w", newline='') as f:
             b_csv = csv.writer(f)
             b_csv.writerows(sess.run(prediction, feed_dict={Net_Input: test_input_Normalization}))









