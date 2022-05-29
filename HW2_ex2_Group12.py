# DISCLAIMER: the script is supposed to be run on TensorFlow 2.3.0, otherwise it will not work.

# import the needed libraries
import argparse
import numpy as np
from sklearn import preprocessing
from scipy.io import wavfile
import scipy.signal as signal
import tensorflow as tf
import os
import pandas as pd

parser = argparse.ArgumentParser()
# receive the version as a keyword argument
parser.add_argument('--version', type=str, required=True,
        help='version can be a, b or c')
args = parser.parse_args()

# set parameters according to each version
if(args.version == "a"):
    alpha=0.75
    
if(args.version == "b"):
    alpha=0.50
    
if(args.version == "c"):
    alpha=0.25

# get the list of sample paths for the 3 sets of train, test and validation
train_files=pd.read_csv('./kws_train_split.txt', sep="\n")
train_files=np.array(train_files.values)
train_files=train_files.squeeze()

test_files=pd.read_csv('./kws_test_split.txt', sep="\n")
test_files=np.array(test_files.values)
test_files=test_files.squeeze()

val_files=pd.read_csv('./kws_val_split.txt', sep="\n")
val_files=np.array(val_files.values)
val_files=val_files.squeeze()

# get the labels from textual file
with open('./labels.txt') as f:
    LABELS = eval(f.read())
    
# download the dataset from the web, unzip it and put it in the suited folder 'data'
zip_path = tf.keras.utils.get_file(origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                                   fname='mini_speech_commands.zip',
                                   extract=True,
                                   cache_dir='.', cache_subdir='data')

# class that transforms wav samples into MFCCs: this is taken from 'lab3', but we added the option to undersample recordings
class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None, mfcc=False):
        
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        num_spectrogram_bins = (frame_length) // 2 + 1

        if mfcc is True:
            # we can compute this matrix only once, to save computation time
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                    self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft
            
    # resample function, through scipy method
    def res(self, audio):
        audio=signal.resample_poly(audio,self.sampling_rate,16000)
        return np.array(audio,dtype=np.float32)

    # passing resampling to tensorflow, as a numpy function    
    def tf_function(self, audio):
        audio = tf.numpy_function(self.res, [audio], tf.float32)
        return audio
    
    # read, decode, resample the .wav file and its related label
    def read(self, file_path):
        # split the path in its components
        parts = tf.strings.split(file_path, os.path.sep)
        # get the label (string)
        label = parts[-2]
        # convert the label to Int value
        label_id = tf.argmax(label == self.labels)
        # read, decode, resample, reshape the audio
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)        
        audio = self.tf_function(audio)        
        audio = tf.squeeze(audio, axis=1)
        return audio, label_id
    
    # utility function (extends the duration of audios shorter than 1s, to keep the sample dimensions steady)
    def pad(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])
        return audio
    
    # cut audio in chuncks, then transform them into spectrograms through stft
    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)
        return spectrogram
    
    # generate MFCCs from the given spectrogram
    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]
        return mfccs

    # design the preprocessing pipeline if we chose to work with spectrograms
    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])
        return spectrogram, label

    # design the preprocessing pipeline if we chose to work with MFCCs
    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)
        return mfccs, label

    # apply the pipeline to each of the files contained in the provided paths list; then batch, shuffle and cache the obtained DS
    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.batch(32)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)
        return ds

# declare a serie of parameters needed for the preprocessing step
options = {'frame_length': 320, 'frame_step': 160, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 10,
        'num_coefficients': 40}
sampling_rate = 8000

# declare a serie of parameters needed to build architectures
strides = [2, 1]
units = len(LABELS)

# Create an instance of the SignalGenerator
generator = SignalGenerator(LABELS, sampling_rate, **options)
# Create Train set
train_ds = generator.make_dataset(train_files, True)
# Create Validation set
val_ds = generator.make_dataset(val_files, False)
# Create Test set
test_ds = generator.make_dataset(test_files, False)

## Callbacks:
## EarlyStopping on the Validation Set metric (stop when it stops improving)
earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_sparse_categorical_accuracy',
    patience=30,
    verbose=1,
    mode='auto',
)

## Learning Rate Scheduling on the Validation set metric (reducing it on plateau)
reduceLR = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_sparse_categorical_accuracy',
    factor=0.3,
    patience=10,
    min_lr=1e-05,
    verbose=1
)

## Checkpoints on the Validation set metric to save the overall best model for the deployment
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    './ds_cnn/chkp_best',
    save_best_only=True,
    monitor = 'val_sparse_categorical_accuracy',
    save_weights_only=True,
    save_freq='epoch'
)

# Build the DS-CNN model
ds_convolutional=tf.keras.Sequential([
    tf.keras.layers.Reshape((49,10,1)),
    tf.keras.layers.Conv2D(alpha*256,3,strides=strides, use_bias=False),
    tf.keras.layers.BatchNormalization(momentum=0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, use_bias=False),
    tf.keras.layers.Conv2D(alpha*256,1,strides=1, use_bias=False),
    tf.keras.layers.BatchNormalization(momentum=0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, use_bias=False),
    tf.keras.layers.Conv2D(alpha*256,1,strides=1, use_bias=False),
    tf.keras.layers.BatchNormalization(momentum=0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units)
    ])

# Compile the model with the given optimizer, loss, metric
ds_convolutional.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) ,metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
ds_convolutional.build((None, 49,10,1))
# Summary of the model's layers
ds_convolutional.summary()
# Training phase
history = ds_convolutional.fit(train_ds,validation_data=val_ds, epochs=300, callbacks=[cp_callback, earlystop, reduceLR])
# Load the best weights from Checkpoint
ds_convolutional.load_weights('./ds_cnn/chkp_best') 

# Prepare to save the model in tflite
run_model = tf.function(lambda x: ds_convolutional(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 49, 10, 1], tf.float32))
ds_convolutional.save('_saved_model_dir/', signatures=concrete_func)

# Create and Save the TfLite Model, after pruning
# Path of the TfLite Model
tflite_model_dir=f"Group12_kws_{args.version}.tflite"
# Convert the model in TfLite
converter = tf.lite.TFLiteConverter.from_saved_model("_saved_model_dir/")
# int8 Quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Serialize Model
tflite_model = converter.convert()
# Compress and Save the TfLite Model
with open(tflite_model_dir, 'wb') as fp:
    fp.write(tflite_model)

