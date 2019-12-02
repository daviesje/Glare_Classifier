import numpy as np
from PIL import Image
from keras.utils import to_categorical

#Image.MAX_IMAGE_PIXELS = None
NET_INPUT = np.array([32,32,3])

def set_inshape(inshape):
    global NET_INPUT
    NET_INPUT = inshape
    return NET_INPUT

def blockshaped(arr,subh,subw):
    h,w = arr.shape[:2]
    return (arr.reshape(h//subh,subh,-1,subw)
            .swapaxes(1,2)
            .reshape(-1,subh,subw))

def read_image(imfile):
    reader = Image.open(imfile,mode='r')
    raster = np.array(reader)
    return raster

def setup_dataset(raster,ratio,shuffle=True):
    #NB: crops image to nearest NET_INPUT pixels
    inshape = raster.shape[:2]//NET_INPUT[:2]
    inpx = inshape*NET_INPUT[:2]
    cropped = raster[:inpx[0],:inpx[1],...]

    #print(cropped.shape,inshape,NET_INPUT[:2]*inshape)

    fulldata = np.zeros(np.append(np.product(inshape),NET_INPUT),dtype=int)
    fulldata[...,0] = blockshaped(cropped[...,0],NET_INPUT[0],NET_INPUT[1])
    fulldata[...,1] = blockshaped(cropped[...,1],NET_INPUT[0],NET_INPUT[1])
    fulldata[...,2] = blockshaped(cropped[...,2],NET_INPUT[0],NET_INPUT[1])

    fulldata = fulldata.astype(float)/255.

    if(shuffle):
        np.random.seed(222)
        np.random.shuffle(fulldata)

    ntrain = int(fulldata.shape[0]*ratio)
    X_train = fulldata[:ntrain]
    X_test = fulldata[ntrain:]

    print(f'Image sliced from {raster.shape} to {X_train.shape} + {X_test.shape}')
    return X_train,X_test

#labels for image classifier
def setup_labels(mask,ratio,return_masks=False,shuffle=True):
    #NB: crops image to nearest 128 pixels
    inshape = mask.shape[:2]//NET_INPUT[:2]
    inpx = inshape*NET_INPUT[:2]
    cropped = mask[:inpx[0],:inpx[1],...]

    #print(cropped.shape,inshape,NET_INPUT[:2]*inshape)

    fulldata = np.zeros(np.append(np.product(inshape),NET_INPUT),dtype=int)
    fulldata[...,0] = blockshaped(cropped[...,0],NET_INPUT[0],NET_INPUT[1])
    fulldata[...,1] = blockshaped(cropped[...,1],NET_INPUT[0],NET_INPUT[1])
    fulldata[...,2] = blockshaped(cropped[...,2],NET_INPUT[0],NET_INPUT[1])

    if(shuffle):
        np.random.seed(222)
        np.random.shuffle(fulldata)

    ntrain = int(fulldata.shape[0]*ratio)
    Y_train = fulldata[:ntrain,...]
    Y_test = fulldata[ntrain:,...]

    #set flag if ANY pixel in the mask is white
    y_train = np.any(Y_train > 200,axis=(1,2,3))
    y_test = np.any(Y_test > 200,axis=(1,2,3))

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print(f'mask sliced from {mask.shape} to {y_train.shape} + {y_test.shape}')
    if return_masks:
        return y_train,y_test,Y_train,Y_test
    else:
        return y_train, y_test
