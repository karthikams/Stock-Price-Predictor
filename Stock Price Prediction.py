
# coding: utf-8

# In[101]:


import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import time

# split data in 80%/10%/10% train/validation/test sets
valid_set_size_percentage = 10 
test_set_size_percentage = 10 


# In[102]:


import quandl
import pandas as pd

# add quandl API key for unrestricted
quandl.ApiConfig.api_key = '5ybZhPgMUSgmzT7BNYmb'
# get the table for daily stock prices and,
# filter the table for selected tickers, columns within a time range
# set paginate to True because Quandl limits tables API to 10,000 rows per call
df = quandl.get_table('WIKI/PRICES', ticker = ['AAPL'], 
                        qopts = { 'columns': ['ticker', 'date', 'open', 'close','low', 'high', 'volume'] }, 
                        date = { 'gte': '2000-01-01', 'lte': '2018-10-14' }, 
                        paginate=True)


# In[103]:


df = df.set_index('date')


# In[104]:


df = df.sort_values(by = 'date')


# In[105]:


df.head(10)


# In[106]:


df.describe()


# In[107]:


import matplotlib.pyplot as plt


# In[108]:


df.info()


# In[109]:


plt.plot(df['open'])
plt.title('Distribution of Open values')
plt.xlabel('year')
plt.ylabel('open')
plt.savefig('open.png')
df['open']


# In[110]:


plt.plot(df['close'])
plt.title('Distribution of Close values')
plt.xlabel('year')
plt.ylabel('close')
plt.savefig('close.png')
df['close']


# In[111]:


plt.plot(df['low'])
plt.title('Distribution of Low values')
plt.xlabel('year')
plt.ylabel('low')
plt.savefig('low.png')


# In[112]:


plt.plot(df['high'])
plt.xlabel('year')
plt.ylabel('high')
plt.title('Distribution of High values')
plt.savefig('high.png')


# In[113]:


plt.plot(df['volume'])
plt.xlabel('year')
plt.ylabel('volume')
plt.title('Distribution of Volume values')
plt.savefig('volume.png')


# In[114]:


plt.figure(figsize=(15, 5));
plt.plot(df[df.ticker == 'AAPL'].open.values, color='black', label='open', linewidth = .2)
plt.plot(df[df.ticker == 'AAPL'].close.values, color='green', label='close')
plt.plot(df[df.ticker == 'AAPL'].low.values, color='blue', label='low')
plt.plot(df[df.ticker == 'AAPL'].high.values, color='magenta', label='high')
plt.title('Stock Price Distribution before Normalization')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend(loc='best')
plt.savefig("before norm.png")


# In[115]:


plt.plot(df[df.ticker == 'AAPL'].volume.values, color='violet', label='volume')
plt.title('stock volume')
plt.xlabel('time [days]')
plt.ylabel('volume')
plt.legend(loc='best')


# In[116]:


# function for min-max normalization of stock
def normalize_data(df):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    df['close'] = min_max_scaler.fit_transform(df.close.values.reshape(-1,1))
    #df['volume'] = min_max_scaler.fit_transform(df['volume'].values.reshape(-1,1))
    return df

# function to create train, validation, test data given stock data and sequence length
def load_data(stock, seq_len):
    data_raw = stock.as_matrix() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len): 
        data.append(data_raw[index: index + seq_len])
    
    data = np.array(data);
    valid_set_size = int(np.round(valid_set_size_percentage/100*data.shape[0]));  
    test_set_size = int(np.round(test_set_size_percentage/100*data.shape[0]));
    train_set_size = data.shape[0] - (valid_set_size + test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]
    y_valid = data[train_set_size:train_set_size+valid_set_size,-1,:]
    
    x_test = data[train_set_size+valid_set_size:,:-1,:]
    y_test = data[train_set_size+valid_set_size:,-1,:]
    
    return [x_train, y_train, x_valid, y_valid, x_test, y_test]

# choose one stock
df_stock = df[df.ticker == 'AAPL'].copy()
df_stock.drop(['ticker'],1,inplace=True)
df_stock.drop(['volume'],1,inplace=True)

cols = list(df_stock.columns.values)
print('df_stock.columns.values = ', cols)

# normalize stock
df_stock_norm = df_stock.copy()
df_stock_norm = normalize_data(df_stock_norm)

# create train, test data
seq_len = 20 # choose sequence length
x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df_stock_norm, seq_len)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_valid.shape = ',x_valid.shape)
print('y_valid.shape = ', y_valid.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ',y_test.shape)


# In[124]:


plt.figure(figsize=(15, 5));
plt.plot(df_stock_norm.open.values, color='violet', label='open')
plt.plot(df_stock_norm.close.values, color='green', label='close')
plt.plot(df_stock_norm.low.values, color='blue', label='low')
plt.plot(df_stock_norm.high.values, color='red', label='high')
#plt.plot(df_stock_norm.volume.values, color='gray', label='volume')
plt.title('Stock Price after Normalization')
plt.xlabel('time [days]')
plt.ylabel('Normalized Price')
plt.legend(loc='best')
plt.savefig("after norm.png")
plt.show()


# In[118]:


## Basic Cell RNN in tensorflow
import time
import csv

mse_values = []
epoch_values = []
index_in_epoch = 0;
perm_array  = np.arange(x_train.shape[0])
np.random.shuffle(perm_array)


# function to get the next batch
def get_next_batch(batch_size):
    global index_in_epoch, x_train, perm_array   
    start = index_in_epoch
    index_in_epoch += batch_size
    
    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array) # shuffle permutation array
        start = 0 # start next epoch
        index_in_epoch = batch_size
        
    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]

# parameters
n_steps = seq_len-1
n_inputs = 4 
n_neurons = 200
n_outputs = 4
n_layers = 4
learning_rate = 0.001
batch_size = 50
n_epochs = 100
train_set_size = x_train.shape[0]
test_set_size = x_test.shape[0]

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

# use Basic RNN Cell
layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)
          for layer in range(n_layers)]

# use Basic LSTM Cell 
#layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.elu)
#          for layer in range(n_layers)]

# use LSTM Cell with peephole connections
#layers = [tf.contrib.rnn.LSTMCell(num_units=n_neurons, 
#                                  activation=tf.nn.leaky_relu, use_peepholes = True)
#          for layer in range(n_layers)]

# use GRU cell
#layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.leaky_relu)
#          for layer in range(n_layers)]
                                                                     
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons]) 
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
outputs = outputs[:,n_steps-1,:] # keep only last output of sequence

loss = tf.reduce_mean(tf.square(outputs - y)) # loss function = mean squared error 
tf.summary.histogram("losss",loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) 
training_op = optimizer.minimize(loss)
loss_summary = tf.summary.scalar("loss", loss)
                                              
# run graph
start_time = time.time()
with tf.Session() as sess: 
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    train_summary_op = tf.summary.merge([loss_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    eval_summary_op = tf.summary.merge([loss_summary])
    eval_summary_dir = os.path.join(out_dir, "summaries", "eval")
    eval_summary_writer = tf.summary.FileWriter(eval_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    sess.run(tf.global_variables_initializer())
    for iteration in range(int(n_epochs*train_set_size/batch_size)):
        x_batch, y_batch = get_next_batch(batch_size) # fetch the next training batch 
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        _, summaries = sess.run([training_op, train_summary_op], feed_dict={X: x_batch, y: y_batch},options=run_options, run_metadata=run_metadata)
        train_summary_writer.add_run_metadata(run_metadata, 'step%d' % iteration)
        train_summary_writer.add_summary(summaries)
        train_summary_writer.flush()
        
        if iteration % int(5*train_set_size/batch_size) == 0:
            time_str = datetime.datetime.now().isoformat()
            mse_train = loss.eval(feed_dict={X: x_train, y: y_train}) 
            mse_valid = loss.eval(feed_dict={X: x_valid, y: y_valid}) 
            print('%.2f epochs: MSE train/valid = %.6f/%.6f %s ms'%(
                iteration*batch_size/train_set_size, mse_train, mse_valid, time_str))
            
            #writing MSE values during training to Excel sheet
            mse_values.append(mse_train)
            epoch_values.append(iteration*batch_size/train_set_size)
            
    with open("MSE", "w") as file:
        writer = csv.writer(file, delimiter=',', dialect = 'excel')
        for item in mse_values:
            writer.writerow([item])
    
    y_train_pred = sess.run(outputs, feed_dict={X: x_train})
    y_valid_pred = sess.run(outputs, feed_dict={X: x_valid})
    y_test_pred = sess.run(outputs, feed_dict={X: x_test})
    
    with open("test_predictions.csv",'w') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerows(y_test_pred)
    
end_time = time.time() - start_time
print("time taken ", end_time)


# In[119]:


#plotting values
plt.figure(figsize=(15, 5))
plt.plot(epoch_values, mse_values, color='red', linewidth = 4)
plt.title('mse vs epoch')
plt.xlabel('epoch')
plt.ylabel('mse')
plt.savefig('mse.png')


# In[120]:


y_train.shape


# In[121]:


y_test_pred


# In[122]:


ft = 0 # 0 = open, 1 = close, 2 = highest, 3 = lowest

## show predictions
plt.figure(figsize=(15, 5));
#plt.subplot(1,2,1);

plt.plot(np.arange(y_train.shape[0]), y_train[:,ft], color='blue', label='train target')

plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_valid.shape[0]), y_valid[:,ft],
         color='gray', label='valid target')

plt.plot(np.arange(y_train.shape[0]+y_valid.shape[0],
                   y_train.shape[0]+y_test.shape[0]+y_test.shape[0]),
         y_test[:,ft], color='black', label='test target')

plt.plot(np.arange(y_train_pred.shape[0]),y_train_pred[:,ft], color='red',
         label='train prediction')

plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_valid_pred.shape[0]),
         y_valid_pred[:,ft], color='orange', label='valid prediction')

plt.plot(np.arange(y_train_pred.shape[0]+y_valid_pred.shape[0],
                   y_train_pred.shape[0]+y_valid_pred.shape[0]+y_test_pred.shape[0]),
         y_test_pred[:,ft], color='green', label='test prediction')

plt.title('past and future stock prices for AAPL')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best');
plt.savefig("overview.png")

#plt.subplot(1,2,2);


# In[123]:


plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_test.shape[0]),
         y_test[:,ft], color='black', label='test target')

plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_test_pred.shape[0]),
         y_test_pred[:,ft], color='green', label='test prediction')

plt.title('future stock prices for AAPL')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best')
plt.savefig("test.png")

corr_price_development_train = np.sum(np.equal(np.sign(y_train[:,1]-y_train[:,0]),
            np.sign(y_train_pred[:,1]-y_train_pred[:,0])).astype(int)) / y_train.shape[0]
corr_price_development_valid = np.sum(np.equal(np.sign(y_valid[:,1]-y_valid[:,0]),
            np.sign(y_valid_pred[:,1]-y_valid_pred[:,0])).astype(int)) / y_valid.shape[0]
corr_price_development_test = np.sum(np.equal(np.sign(y_test[:,1]-y_test[:,0]),
            np.sign(y_test_pred[:,1]-y_test_pred[:,0])).astype(int)) / y_test.shape[0]

print('correct sign prediction for close - open price for train/valid/test: %.2f/%.2f/%.2f'%(
    corr_price_development_train, corr_price_development_valid, corr_price_development_test))

