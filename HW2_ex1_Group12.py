# import the needed libraries
import zlib
import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot

parser= argparse.ArgumentParser()
# receive the version as a keyword argument
parser.add_argument('--version', type=str, required=True,
        help='version can be a or b')
args = parser.parse_args()
# set parameters according to each version
if(args.version == "a"):
    n_output=3
    f_sparsity=0.8
    
if(args.version == "b"):
    n_output=9
    f_sparsity=0.85
        
# set seed
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# read dataset
df = pd.read_csv("jena_climate_2009_2016.csv")
column_indices = [2, 5]    ## Consider just Temperature and Humidity
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

# defining train, test and validation set
n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

# compute mean and standard deviation on the Train Set
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

# set parameters
input_width = 6    ## Num of elements for each windows
output_step = n_output
LABEL_OPTIONS = 2    ## Each window considers both measures for Temperature and Humidity

# Preprocessing and Data Preparation - Windows Generator 
class WindowGenerator:
    def __init__(self, input_width, label_options, mean, std,output_step):
        self.input_width = input_width
        self.label_options = label_options
        self.output_step = output_step
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])    
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

    ## Split Windows according to the input width and number of output steps
    def split_window(self, features):
        inputs = features[:, :-self.output_step, :]

        if self.label_options < 2:
            labels = features[:, -self.output_step:, self.label_options]
            labels = tf.expand_dims(labels, -1)
            num_labels = 1
        else:
            labels = features[:, -self.output_step:, :]
            num_labels = 2

        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None,self.output_step, num_labels])

        return inputs, labels

    ## Normalization of each window with mean and std of the training set
    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    ## Preprocess pipeline
    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    ## Dataset Creation
    def make_dataset(self, data, train):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=input_width+output_step,
                sequence_stride=1,
                batch_size=32)
        ds = ds.map(self.preprocess)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds


## Create an instance of the WindowGenerator
generator = WindowGenerator(input_width, LABEL_OPTIONS, mean, std,output_step)
## Create Train set
train_ds = generator.make_dataset(train_data, True)
## Create Validation set
val_ds = generator.make_dataset(val_data, False)
## Create Test set
test_ds = generator.make_dataset(test_data, False)    


## MAE metrics of temperature and humidity for multi-output models
## Temperature MAE
class My_mae_t(tf.keras.metrics.Metric):
    
    def __init__(self, name='My_mae_t', **kwargs):
        super(My_mae_t, self).__init__(name=name, **kwargs)
        self.mae_temp = self.add_weight(name='mae_temp', initializer='zeros') # counter that sums the mean absolute errors over temperature for a single prediction
        self.count = self.add_weight(name='count', initializer='zeros') # counter that counts the number of predictions
        
    # update the counters (at each new prediction)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = y_pred[:,:,0] # select involved values
        y_true = y_true[:,:,0] # select involved values
        
        loss = abs(y_true - y_pred) # calculate the absolute distance
        
        # add values to counters
        self.mae_temp.assign_add(tf.reduce_mean(loss)) 
        self.count.assign_add(1)
    
    # return final MAE value        
    def result(self):
        return self.mae_temp/self.count
    
    # reset MAE state
    def reset_state(self):
        self.mae_temp.assign(0)
        self.count.assign(0)
        
## Humidity MAE
class My_mae_h(tf.keras.metrics.Metric):
    
    def __init__(self, name='My_mae_h', **kwargs):
        super(My_mae_h, self).__init__(name=name, **kwargs)
        self.mae_hum = self.add_weight(name='mae_hum', initializer='zeros') # counter that sums the mean absolute errors over humidity for a single prediction
        self.count = self.add_weight(name='count', initializer='zeros') # counter that counts the number of predictions
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = y_pred[:,:,1] # select involved values
        y_true = y_true[:,:,1] # select involved values
        
        loss = abs(y_true - y_pred) # calculate the absolute distance

        # add values to counters
        self.mae_hum.assign_add(tf.reduce_mean(loss)) 
        self.count.assign_add(1)
        
    # return final MAE value 
    def result(self):
        return self.mae_hum/self.count
    
    # reset MAE state
    def reset_state(self):
        self.mae_hum.assign(0)
        self.count.assign(0)


## Callbacks:
## EarlyStopping on the Validation Set loss (stop when it stops improving)
earlystop = tf.keras.callbacks.EarlyStopping(
  monitor='val_loss',
  patience=20,
  verbose=1,
  mode='auto',
)

## Learning Rate Scheduling on the Validation set Loss (reducing it on plateau)
reduceLR = tf.keras.callbacks.ReduceLROnPlateau(
  monitor='val_loss',
  factor=0.3,
  patience=10,
  min_lr=1e-05,
  verbose=1
)

## Checkpoints on the Validation set loss to save the overall best model for the deployment
model_save = tf.keras.callbacks.ModelCheckpoint("best_model.hdf5", monitor='val_loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

## Build of mlp model
mlp=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(6, 2)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(output_step*2),
    tf.keras.layers.Reshape((output_step,2))
    ])

## Instance creation of both temperature and humidity MAEs
temp_mae = My_mae_t()
hum_mae = My_mae_h()

## Compile the model with adam optimizer, MSE as loss function, temperature and humidity MAEs to monitor
mlp.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=[temp_mae, hum_mae])

## Train the model on Train set evaluating the loss on Validation set, using predefined callbacks 
history = mlp.fit(train_ds,validation_data=val_ds, epochs=300,callbacks=[earlystop,reduceLR,model_save])

## Summary of the model's layers
print(mlp.summary())

## Load the best weights from Checkpoint
mlp.load_weights('best_model.hdf5')
    
## Magnitude-Based Pruning Schedule:
## Sparsity Scheduler with Polynomial Decay, change of sparsity enforced during training between initial and final values
pruning_params = {'pruning_schedule':
    tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.50,
    final_sparsity=f_sparsity,
    begin_step=len(train_ds)*0,
    end_step=len(train_ds)*15)
    }

## Create a Model with Pruning Functionality
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
model_mlp = prune_low_magnitude(mlp, **pruning_params)

## Callback for Magnitude-Based Pruning on the re-trained model 
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    
## Train the model
model_mlp.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[temp_mae, hum_mae])
input_shape = [32, 6, 2]
model_mlp.build(input_shape)
model_mlp.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=callbacks)  

## Remove Additional Data added by prune_low_magnitude
model_mlp = tfmot.sparsity.keras.strip_pruning(model_mlp)

## Prepare to save the Model in tflite
run_model = tf.function(lambda x: model_mlp(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 6, 2], tf.float32))
model_mlp.save('_saved_model_dir_prun_mlp/', signatures=concrete_func)

## Create and Save the TfLite Model, after pruning, Int8-PTQ and compression
## Path of the TfLite Model
tflite_model_dir=f"Group12_th_{args.version}.tflite.zlib"
## Convert the model in TfLite
converter = tf.lite.TFLiteConverter.from_saved_model("_saved_model_dir_prun_mlp/")
## int8 Quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
## Serialize Model
tflite_model = converter.convert()

## Compress and Save the TfLite Model (zlib format)
with open(tflite_model_dir, 'wb') as fp:
    tflite_compressed = zlib.compress(tflite_model)
    fp.write(tflite_compressed)

    


# In[ ]:




