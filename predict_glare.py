import numpy as np
from keras.datasets import mnist
from keras.models import load_model

from matplotlib import pyplot as plt
from matplotlib import gridspec

import sys

import dset
import models

image_fn = sys.argv[1]
model_name = sys.argv[2]
suffix = sys.argv[3]

inshape = dset.set_inshape(np.array([32,32,3]))

#this is a really roundabout way of joining the datasets using existing functions
#I should specify a new one that doesn't split
ratio = 0.9
image = dset.read_image(image_fn)
X_train,X_test = dset.setup_dataset(image,ratio,shuffle=False)
X_data = np.append(X_train,X_test,axis=0)

model = load_model(f'./models/{model_name}.hdf5')

#classify new image
predictions = np.argmax(model.predict(X_data),axis=1)

#reshape to output square grid
outshape = np.array([np.sqrt(len(predictions)),np.sqrt(len(predictions))]).astype(int)
predictions = predictions.reshape(outshape)
#convert to RGB with white as glare tiles
predictions = (np.tile(predictions[:,:,None],(1,1,3))*255).astype(int)

np.save(f'./predictions_{model_name}_{suffix}.npy',predictions)

ext = [0,image.shape[0],0,image.shape[1]]
gs = gridspec.GridSpec(1,2)
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(gs[0])
ax.imshow(image,extent=ext)
ax.set_title('Image')
ax = fig.add_subplot(gs[1],sharex=ax,sharey=ax)
ax.set_title('ML Predictions')
ax.imshow(predictions,extent=ext)

fig.savefig(f'./predictions_{model_name}_{suffix}.png')
plt.show()
