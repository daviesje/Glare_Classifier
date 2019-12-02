import numpy as np
from PIL import Image
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D,Dropout
from keras.utils import to_categorical

from matplotlib import pyplot as plt
from matplotlib import gridspec

Image.MAX_IMAGE_PIXELS = None
NET_INPUT = np.array([32,32,3])
TRAIN,TEST = 0.75,0.25

def blockshaped(arr,subh,subw):
    h,w = arr.shape[:2]
    return (arr.reshape(h//subh,subh,-1,subw)
            .swapaxes(1,2)
            .reshape(-1,subh,subw))

def read_image(imfile):
    reader = Image.open(imfile,mode='r')
    raster = np.array(reader)
    return raster

def setup_dataset(raster,ratio):
    #NB: crops image to nearest NET_INPUT pixels
    inshape = raster.shape[:2]//NET_INPUT[:2]
    inpx = inshape*NET_INPUT[:2]
    cropped = raster[:inpx[0],:inpx[1],...]

    #print(cropped.shape,inshape,NET_INPUT[:2]*inshape)

    fulldata = np.zeros(np.append(np.product(inshape),NET_INPUT),dtype=int)
    fulldata[...,0] = blockshaped(cropped[...,0],NET_INPUT[0],NET_INPUT[1])
    fulldata[...,1] = blockshaped(cropped[...,1],NET_INPUT[0],NET_INPUT[1])
    fulldata[...,2] = blockshaped(cropped[...,2],NET_INPUT[0],NET_INPUT[1])

    np.random.seed(222)
    np.random.shuffle(fulldata)

    ntrain = int(fulldata.shape[0]*ratio)
    X_train = fulldata[:ntrain]
    X_test = fulldata[ntrain:]

    print(f'Image sliced from {raster.shape} to {fulldata.shape}, {ntrain} for training')
    return X_train,X_test

#labels for image classifier
def setup_labels(mask,ratio):
    #NB: crops image to nearest 128 pixels
    inshape = mask.shape[:2]//NET_INPUT[:2]
    inpx = inshape*NET_INPUT[:2]
    cropped = mask[:inpx[0],:inpx[1],...]

    #print(cropped.shape,inshape,NET_INPUT[:2]*inshape)

    fulldata = np.zeros(np.append(np.product(inshape),NET_INPUT),dtype=int)
    fulldata[...,0] = blockshaped(cropped[...,0],NET_INPUT[0],NET_INPUT[1])
    fulldata[...,1] = blockshaped(cropped[...,1],NET_INPUT[0],NET_INPUT[1])
    fulldata[...,2] = blockshaped(cropped[...,2],NET_INPUT[0],NET_INPUT[1])

    np.random.seed(222)
    np.random.shuffle(fulldata)

    ntrain = int(fulldata.shape[0]*ratio)
    Y_train = fulldata[:ntrain,...]
    Y_test = fulldata[ntrain:,...]

    y_train = np.any(Y_train > 200,axis=(1,2,3))
    y_test = np.any(Y_test > 200,axis=(1,2,3))

    print(f'mask sliced from {mask.shape} to {Y_train.shape} + {Y_test.shape} ')
    return y_train, y_test, Y_train, Y_test

fullimage = read_image('./tiles/frames/Bleaching_glare_map_big_04_10.png')
X_train,X_test = setup_dataset(fullimage,TRAIN)
labels = read_image('./tiles/masks/Bleaching_glare_polygon_big_04_10.png')
y_train,y_test,Y_train,Y_test = setup_labels(labels,TRAIN)

gs = gridspec.GridSpec(2,10)
fig = plt.figure(figsize=(10,4))
for i in range(10):
    ax = fig.add_subplot(gs[0,i])
    ax.imshow(X_train[i*20+500])
    ax = fig.add_subplot(gs[1,i])
    ax.imshow(Y_train[i*20+500])
    print(np.amax(Y_train[i*20+500]),y_train[i*20+500])

plt.show()
#download mnist data and split into train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

#reshape data to fit model
#X_train = X_train.reshape(60000,28,28,1)/255.
#X_test = X_test.reshape(10000,28,28,1)/255.

X_train = X_train.astype(float)/255.
X_test = X_test.astype(float)/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(np.sum(y_train,axis=0)/y_train.shape[0])

#create model
model = Sequential()
#add model layers
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

#model.add(Conv2D(128, kernel_size=1, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

nepochs = 30

#train the model
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=nepochs)
model.save('model_32_test.hdf5')

plt.figure(figsize=(12,8))
plt.plot(np.arange(0, nepochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, nepochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, nepochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, nepochs), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

plt.savefig('./plots/history_32_test.png')


print(f'predictions for first 10: {np.argmax(model.predict(X_test[:30]),axis=1)}')
print(f'data for first 10: {np.argmax(y_test[:30],axis=1)}')
