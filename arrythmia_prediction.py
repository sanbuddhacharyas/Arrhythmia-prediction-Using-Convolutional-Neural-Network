import tensorflow as tf
import numpy as np
import cv2
import glob
from server import Socket
import matplotlib.pyplot as plt
import biosppy

class ecg_prediction():
    def __init__(self):
        self.path ='/home/sansii/Desktop/python/ECG_arrihmia_prediction'
        
        self.model = tf.keras.Sequential()

        self.model.add(tf.keras.layers.Conv2D(64, (3,3),strides = (1,1), input_shape = (128,128,1),kernel_initializer='glorot_uniform'))

        self.model.add(tf.keras.layers.ELU())

        self.model.add(tf.keras.layers.BatchNormalization())

        self.model.add(tf.keras.layers.Conv2D(64, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

        self.model.add(tf.keras.layers.ELU())

        self.model.add(tf.keras.layers.BatchNormalization())

        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides= (2,2)))

        self.model.add(tf.keras.layers.Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

        self.model.add(tf.keras.layers.ELU())

        self.model.add(tf.keras.layers.BatchNormalization())

        self.model.add(tf.keras.layers.Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

        self.model.add(tf.keras.layers.ELU())

        self.model.add(tf.keras.layers.BatchNormalization())

        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides= (2,2)))

        self.model.add(tf.keras.layers.Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

        self.model.add(tf.keras.layers.ELU())

        self.model.add(tf.keras.layers.BatchNormalization())

        self.model.add(tf.keras.layers.Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

        self.model.add(tf.keras.layers.ELU())

        self.model.add(tf.keras.layers.BatchNormalization())

        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides= (2,2)))

        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(2048))

        self.model.add(tf.keras.layers.ELU())

        self.model.add(tf.keras.layers.BatchNormalization())

        self.model.add(tf.keras.layers.Dropout(0.5))

        self.model.add(tf.keras.layers.Dense(8, activation='softmax'))

        self.model.load_weights(self.path+"/weights/weight.h5")

    def make_plot(self):
        signal = np.loadtxt(self.path+"/ecg.txt")
        data = np.delete(signal,np.argwhere(signal > 1000))
        # signal = signal.readlines()
        # signal = [int(i.split('\n')[0]) for i in signal]
        # # for i in signal[0:10]:
        # #     print(int(i.split('\n')[0]))
            
        # print("Length of data " +str(len(signal)))
        # data = np.array(signal)
        
        # print("data",data)
        # data = data.astype('int')
        # print("data after",data)
        signals = []
        self.X_test = []
        count = 1
        peaks =  biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate = 200)[0]
        print(peaks)
        for i in (peaks[1:-1]):
            diff1 = abs(peaks[count - 1] - i)
            diff2 = abs(peaks[count + 1]- i)
            x = peaks[count - 1] + diff1//2
            y = peaks[count + 1] - diff2//2
            signal = data[x:y]
            signals.append(signal)
            count += 1
        signals  = np.array(signals)

        for num, i in enumerate(signals):
            plt.figure(1)
            plt.plot(i,color = 'black')
            plt.axis('off')
            plt.savefig(self.path+"/ecg_images/"+str(num)+".png")
            plt.cla()

        ecg_path = glob.glob(self.path+"/ecg_images/*.png")
        print(len(ecg_path))
        for i in ecg_path:
            image = cv2.imread(i,cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (128,128), interpolation = cv2.INTER_LANCZOS4)
            image = np.array(image)
            image = np.expand_dims(image, axis = 2 )
            image = image.astype('int')
            self.X_test.append(image)


        self.X_test = np.array(self.X_test)  
        print(self.X_test.shape)

        # self.Y_test = np.full((self.X_test.shape[0],), 0)
        # self.Y_test = convert_to_one_hot(self.Y_test,8).T

    def predict(self):
        logits =  self.model.predict(self.X_test)
        return ( np.argmax(logits, axis=1))

host_id = '10.42.0.1'
port = 8080

ecg_pre = ecg_prediction()
soc = Socket(host_id,port)

while True:
    soc.Listen(1)
    print("Waiting for receiving")
    soc.Received(1024)   
    ecg_pre.make_plot()
    out = ecg_pre.predict()
    make = b''
    for i in out:
        print(str(i).encode())
        make = make + str(i).encode()+ b","
    soc.Send(make)

# ecg_pre.make_plot()
# print(ecg_pre.predict())

    #

