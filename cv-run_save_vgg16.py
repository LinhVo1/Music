#!/usr/bin/env python
# coding: utf-8


import argparse

parser = argparse.ArgumentParser(description="run",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--image_len", default=32, type=int, help="IMAGE_LEN")
parser.add_argument("-b", "--batch_size", default=32, type=int,  help="BATCH_SIZE")
parser.add_argument("-t", "--folder_train",default='train',  help="folder_train")
parser.add_argument("-r", "--folder_test", default='test',  help="folder test")
parser.add_argument("-e", "--num_epoch", default=3, type=int, help="epoch")
parser.add_argument("-p", "--patience_epoch", default=3, type=int, help="epoch patience")
parser.add_argument("-c", "--num_class", default=2, type=int, help="number of classes")
parser.add_argument("-l", "--learning_rate", default=0.0001, type=float, help="number of classes")
parser.add_argument("-k", "--nkfold", default=0, type=int, help="kfold ")
parser.add_argument("-f", "--folder", default='save_models', help="folder to save")
parser.add_argument("--seed", default=1,type=int, help="folder to save")
parser.add_argument("--mod", default='vgg', help="model")
parser.add_argument("--nfilter", default=64, type=int, help="number of filter (for cnn3)")

args = parser.parse_args()
config = vars(args)
print(config)



# setup cac bien
# IMAGE_LEN = 32
# BATCH_SIZE = 32
# folder_train = 'data-train'
# folder_test = 'data-test'
# num_epoch = 3

# IMAGE_LEN = config.image_len
# BATCH_SIZE = config.batch_size
# folder_train = config.folder_train
# folder_test = config.folder_test
# num_epoch = config.num_epoch

IMAGE_LEN = args.image_len
BATCH_SIZE = args.batch_size
folder_train = args.folder_train
folder_test = args.folder_test
path_2 = folder_train + folder_test
path_named = path_2.replace("/", ".")
num_epoch = args.num_epoch
num_class = args.num_class
learning_rate=args.learning_rate
patience_epoch=args.patience_epoch
nkfold=args.nkfold
folder_save = args.folder
seed_v = args.seed
mod = args.mod

# IMAGE_LEN = args['image_len']
# BATCH_SIZE = args['batch_size']
# folder_train = args['folder_train']
# folder_test = args['folder_test']
# num_epoch = args['num_epoch']


import numpy as np
import os
import PIL
import PIL.Image
import matplotlib.pyplot as plt
#khai báo frame work tensorflow
import tensorflow as tf
import random
#import keras từ frame work tensorflow
from tensorflow import keras
import tensorflow.keras 
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, MaxPool2D
from tensorflow.keras.layers import Conv2D, InputLayer
from tensorflow.keras.layers import MaxPooling2D




# gan cac seed de co kq thuc nghiem giong

tf.random.set_seed(seed_v)
random.seed(seed_v)
np.random.seed(seed_v)

import time
start = time.time()


## tim tap tin de tranh bi trung
import os, fnmatch
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result



#512x512 size của ảnh

IMAGE_SIZE = (IMAGE_LEN, IMAGE_LEN)
#chia dữ liệu huấn luyện/ kiểm thử thành từng batch

#tiền sử lý ảnh
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #link dẫn ảnh
    folder_train,
    #chia chia train và val
    #validation_split=0,
    #subset="training",
    label_mode = "categorical",
    #seed=1,
    #size của ảnh
    image_size=IMAGE_SIZE,
    #batch_size : chỉa ảnh vào từng batch để trainning như trong bài là 32
    batch_size=BATCH_SIZE)
#tiền sử lý ảnh
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #link dẫn ảnh
    folder_test,
    #chia chia train và val
    #validation_split=0,
    #subset="training",
    label_mode = "categorical",
    #seed=1,
    #size của ảnh
    image_size=IMAGE_SIZE,
    #batch_size : chỉa ảnh vào từng batch để trainning như trong bài là 32
    batch_size=BATCH_SIZE)



from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
#from keras.utils import to_categorical
from sklearn import preprocessing
#from exploit_pred import *

def model_effi(num_classes = num_class,   image_size = IMAGE_LEN, batch_size = BATCH_SIZE):


    img_input = Input(shape=(image_size,image_size,3))

    # 1st convolutional block
    x = Conv1D(2, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification layer
    x = Dense(2, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)

    # Logits layer
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='dpacontest_v4')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


if nkfold in [0,1]:

    if mod == 'vgg':
        name_model = ''
    else:
        name_model = mod + '_'
    name_saved= name_model + path_named + '_c'+ str(num_class)+ '_s' + str(IMAGE_LEN) + '_b'+ str(BATCH_SIZE) +  '_e'+ str(num_epoch) +'_p'+ str(patience_epoch) +  '_lr'+str(learning_rate) + '_se'+ str(seed_v) +'_k'+ str(nkfold) + 'nfilter'+str(args.nfilter) 
    print ('name_saved='+name_saved)
    n_files = find(name_saved + '*.json',folder_save )
    if len(n_files)>0:
        #print('name_saved'+name_saved) #dung thuc nghiem neu lam roi
        print('thuc nghiem '+name_saved+' da lam roi!')
        exit()

    if mod == 'vgg':
        model = Sequential()
        # model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(Conv2D(input_shape=(IMAGE_LEN ,IMAGE_LEN,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        # model.add(Conv2D(input_shape=(64 ,64,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=num_class, activation="softmax"))
        from tensorflow.keras.optimizers import Adam
        #opt = Adam(lr=0.0001)
        opt = Adam(lr=learning_rate)
        model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        print('vggvggvggvgg')
        model.summary()
   
    elif mod == 'fc':
        model = Sequential()       
        model.add(InputLayer(input_shape=(IMAGE_LEN ,IMAGE_LEN,3)))  
        model.add(Flatten())
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=num_class, activation="softmax"))
        from tensorflow.keras.optimizers import Adam
        #opt = Adam(lr=0.0001)
        opt = Adam(lr=learning_rate)
        model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        print('fc')
        model.summary()
    elif  mod == 'cnn1':
        model = Sequential()
        model.add(Conv2D(input_shape=(IMAGE_LEN ,IMAGE_LEN,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(Flatten())
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=num_class, activation="softmax"))
        from tensorflow.keras.optimizers import Adam
        #opt = Adam(lr=0.0001)
        opt = Adam(lr=learning_rate)
        model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        print('cnn1')
        model.summary()

    elif  mod == 'cnn2':
        model = Sequential()
        model.add(Conv2D(input_shape=(IMAGE_LEN ,IMAGE_LEN,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        
        model.add(Flatten())
        model.add(Dense(units=num_class, activation="softmax"))
        from tensorflow.keras.optimizers import Adam
        opt = Adam(lr=learning_rate)
        model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        print('cnn2')
        model.summary()
    
    elif  mod == 'cnn3':
        model = Sequential()
        model.add(Conv2D(input_shape=(IMAGE_LEN ,IMAGE_LEN,3),filters=args.nfilter,kernel_size=(3,3),padding="same", activation="relu"))
     
        model.add(Flatten())
        model.add(Dense(units=num_class, activation="softmax"))
        from tensorflow.keras.optimizers import Adam
        opt = Adam(lr=learning_rate)
        model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        print('cnn3')
        model.summary()

    elif  mod == 'efficient':

        # https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
        from tensorflow.keras.applications import EfficientNetB0

        import tensorflow as tf

        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
            print("Device:", tpu.master())
            strategy = tf.distribute.TPUStrategy(tpu)
        except ValueError:
            print("Not connected to a TPU runtime. Using CPU/GPU strategy")
            strategy = tf.distribute.MirroredStrategy()


        from tensorflow.keras.models import Sequential
        from tensorflow.keras import layers

        # img_augmentation = Sequential(
        #     [
        #         layers.RandomCrop(IMAGE_LEN,IMAGE_LEN)
                
        #     ],
        #     name="img_augmentation",
        # )

        with strategy.scope():
            inputs = layers.Input(shape=(IMAGE_LEN, IMAGE_LEN, 3))
            #x = img_augmentation(inputs)
            outputs = EfficientNetB0(include_top=True, weights=None, classes=num_class)(inputs)

            model = tf.keras.Model(inputs, outputs)
            model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )

        model.summary()

        #epochs = 40  # @param {type: "slider", min:10, max:100}
        #hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2)


    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_epoch)   
    history = model.fit(train_ds, validation_data=test_ds, epochs=num_epoch, verbose=1,callbacks=[callback])
    #print('targets[test].shape')
    #print(targets[test].shape)
    #print(targets)
    #history = model.fit(data_full[train], targets[train], validation_data=(data_full[test], targets[test]), epochs=num_epoch, verbose=1,callbacks=[callback])



    from datetime import datetime
    now=datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    #print("date and time:",date_time)

    # name with hyperparameters used
    name_saved=  name_saved +'_' + str(date_time) #+'_k'+str(i)

    #i=i+1
    #model.save('VGG16_new.h5')
    print('Model Saved!')

    model.save(folder_save+'/'+ name_saved +'.h5')
    model_json = model.to_json()
    with open( folder_save + '/' + name_saved +".json", "w") as json_file:
        json_file.write(model_json)

        # luu lai log
    end = time.time()
    print("time run: ", end - start)

    ep_arr=range(1, len(history.history['accuracy'])+1, 1)
    idx = len(history.history['accuracy'])-1 #index of mang
    train_acc = history.history['accuracy']
    val_acc= history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    title_cols = np.array(["ep","train_acc","valid_acc","train_loss","valid_loss"])  
    res=(ep_arr,train_acc,val_acc, train_loss,val_loss)
    res=np.transpose(res)
    combined_res=np.array(np.vstack((title_cols,res)))

    
    log_name1 = name_saved +'s1'
    np.savetxt(folder_save + '/'+log_name1 +".txt", combined_res, fmt="%s",delimiter="\t") 

    #print('val_acc[len(history.history[accuracy])]' ) 
    #print(val_acc[len(history.history['accuracy'])-1])
    
    #log 2 luu lai tham so va cac thong tin ve sample
    log_name2 = name_saved +'s2_'  + 'time'+ str(round(end - start,2)) + 'acc' +str(round(val_acc[idx],3)) 
    #np.savetxt('save_models/'+log_name+"log2.txt", args, fmt="%s",delimiter="\t")
    with open(folder_save + '/'+log_name2+ ".txt", 'w') as f:
        f.write(str(args))
    title_cols = np.array(["samples_train","samples_test","train_acc","train_loss","val_acc","val_loss"])  
    
    
    train_labels = np.concatenate(list(train_ds.map(lambda x, y:y)))
    test_labels = np.concatenate(list(test_ds.map(lambda x, y:y)))
    
    res=(len(train_labels),len(test_labels),train_acc[idx],train_loss[idx],val_acc[idx],val_loss[idx])
    res=np.transpose(res)
    combined_res=np.array(np.vstack((title_cols,res)))

    with open(folder_save+ '/'+log_name2+ ".txt", "ab") as f:
        f.write(b"\n")
        np.savetxt(f, combined_res, fmt="%s",delimiter="\t")     

else:
    # lay label and data
    train_images = np.concatenate(list(train_ds.map(lambda x, y:x)))
    train_labels = np.concatenate(list(train_ds.map(lambda x, y:y)))

    # merge 2 data: train/test lai
    test_images = np.concatenate(list(test_ds.map(lambda x, y:x)))
    test_labels = np.concatenate(list(test_ds.map(lambda x, y:y)))

    data_full = np.concatenate((train_images, test_images), axis=0)
    targets = np.concatenate((train_labels, test_labels), axis=0)
    print('datadatadatadatadatadata')
    print(data_full.shape)
    print(targets.shape)
    #targets=targets.flatten()
    #print(targets)
    from sklearn.model_selection import StratifiedKFold, KFold
    #skf = StratifiedKFold(n_splits=nkfold)
    skf = KFold(n_splits=nkfold)
    skf.get_n_splits(data_full, test_labels)
    # xoa cac bien
    del train_ds
    del test_ds
    del test_images
    del test_labels

    i=1

    for train, test in skf.split(data_full, targets):

        if mod == 'vgg':
            name_model = ''
        else:
            name_model = mod + '_'

        name_saved=  name_model + '_c'+ str(num_class)+ '_s' + str(IMAGE_LEN) + '_b'+ str(BATCH_SIZE) +  '_e'+ str(num_epoch) +'_p'+ str(patience_epoch) +  '_lr'+str(learning_rate) + '_se'+ str(seed_v) +'_k'+ str(nkfold)+ '_'+ str(i)
        i=i+1
        n_files = find(name_saved + '*.json',folder_save )
        if len(n_files)>0:
            print('thuc nghiem '+name_saved+' da lam roi!')
            continue #di den k tiep theo
        
        

        # model = Sequential()
        # # model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        # model.add(Conv2D(input_shape=(IMAGE_LEN ,IMAGE_LEN,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        # # model.add(Conv2D(input_shape=(64 ,64,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        # model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        # model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        # model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        # model.add(Flatten())
        # model.add(Dense(units=4096,activation="relu"))
        # model.add(Dense(units=4096,activation="relu"))
        # model.add(Dense(units=num_class, activation="softmax"))
        # from tensorflow.keras.optimizers import Adam
        # #opt = Adam(lr=0.0001)
        # opt = Adam(lr=learning_rate)
        # model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        # model.summary()


        if mod == 'vgg':
            model = Sequential()
            # model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
            model.add(Conv2D(input_shape=(IMAGE_LEN ,IMAGE_LEN,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
            # model.add(Conv2D(input_shape=(64 ,64,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
            model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Flatten())
            model.add(Dense(units=4096,activation="relu"))
            model.add(Dense(units=4096,activation="relu"))
            model.add(Dense(units=num_class, activation="softmax"))
            from tensorflow.keras.optimizers import Adam
            #opt = Adam(lr=0.0001)
            opt = Adam(lr=learning_rate)
            model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
            print('vggvggvggvgg')
            model.summary()
    
        elif mod == 'ef':
            print('efefefefefef')
            model = model_effi(num_classes = num_class,   image_size = IMAGE_LEN, batch_size = BATCH_SIZE)
            model.summary()
            
            
        elif mod == 'vgg5':
            model = Sequential()
            # model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
            model.add(Conv2D(input_shape=(IMAGE_LEN ,IMAGE_LEN,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
            # model.add(Conv2D(input_shape=(64 ,64,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
            model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
            model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
            model.add(Flatten())
            model.add(Dense(units=4096,activation="relu"))
            model.add(Dense(units=4096,activation="relu"))
            model.add(Dense(units=4096,activation="relu"))
            model.add(Dense(units=4096,activation="relu"))
            model.add(Dense(units=4096,activation="relu"))
            model.add(Dense(units=num_class, activation="softmax"))
            from tensorflow.keras.optimizers import Adam
            #opt = Adam(lr=0.0001)
            opt = Adam(lr=learning_rate)
            model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
            print('vggvggvggvgg')
            model.summary()

        
        #history = model.fit(data_full[train], targets[train],
        #            batch_size=batch_size,
        #            epochs=2)
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_epoch)   
        #history = model.fit(train_ds, validation_data=test_ds, epochs=num_epoch, verbose=1,callbacks=[callback])
        #print('targets[test].shape')
        #print(targets[test].shape)
        #print(targets)
        history = model.fit(data_full[train], targets[train], validation_data=(data_full[test], targets[test]), epochs=num_epoch, verbose=1,callbacks=[callback])

        from datetime import datetime
        now=datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")
        #print("date and time:",date_time)

        # save model
        #name_saved= 'class'+ str(num_class) +  'VGG16_e'+ str(num_epoch) +'p'+ str(patience_epoch)+ '_s' + str(IMAGE_LEN) +  '_lr'+str(learning_rate) +'_k'+str(i)
        #i=i+1
        #model.save('VGG16_new.h5')
        print('Model Saved!')

        model.save(folder_save+'/'+ name_saved+ '_'+ str(date_time) + '.h5')
        model_json = model.to_json()
        with open( folder_save + '/' + name_saved + '_'+ str(date_time) + ".json", "w") as json_file:
            json_file.write(model_json)


            # luu lai log
        end = time.time()
        print("time run: ", end - start)

        ep_arr=range(1, len(history.history['accuracy'])+1, 1)
        idx = len(history.history['accuracy'])-1 #index of mang
        train_acc = history.history['accuracy']
        val_acc= history.history['val_accuracy']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        title_cols = np.array(["ep","train_acc","valid_acc","train_loss","valid_loss"])  
        res=(ep_arr,train_acc,val_acc, train_loss,val_loss)
        res=np.transpose(res)
        combined_res=np.array(np.vstack((title_cols,res)))

        
       
        log_name1 = name_saved +'s1' + '_'+ str(date_time)
        np.savetxt(folder_save + '/'+log_name1+ ".txt", combined_res, fmt="%s",delimiter="\t") 

        #print('val_acc[len(history.history[accuracy])]' ) 
        #print(val_acc[len(history.history['accuracy'])-1])
        
        #log 2 luu lai tham so va cac thong tin ve sample
        log_name2 = name_saved +'s2' + '_'+ str(date_time) + 't'+ str(round(end - start,2)) + 'acc' +str(round(val_acc[idx],3)) 
        #np.savetxt('save_models/'+log_name+"log2.txt", args, fmt="%s",delimiter="\t")
        with open(folder_save + '/'+log_name2+".txt", 'w') as f:
            f.write(str(args))
        title_cols = np.array(["samples_train","samples_test","train_acc","train_loss","val_acc","val_loss"])  
        res=(len(targets[train]),len(targets[test]),train_acc[idx],train_loss[idx],val_acc[idx],val_loss[idx])
        res=np.transpose(res)
        combined_res=np.array(np.vstack((title_cols,res)))

        with open(folder_save+ '/'+log_name2 + ".txt", "ab") as f:
            f.write(b"\n")
            np.savetxt(f, combined_res, fmt="%s",delimiter="\t")     