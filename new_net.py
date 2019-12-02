import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D,Dropout
from keras.utils import to_categorical

from matplotlib import pyplot as plt
from matplotlib import gridspec
import dset
import models
import sys

image_fn = sys.argv[1]
labels_fn = sys.argv[2]
model_name = sys.argv[3]

#default to 30 epochs
if len(sys.argv) > 3:
    nepochs = int(sys.argv[4])
else:
    nepochs = 30

inshape = dset.set_inshape(np.array([32,32,3]))

ratio = 0.75
image = dset.read_image(image_fn)
labels = dset.read_image(labels_fn)
X_train,X_test = dset.setup_dataset(image,ratio)
y_train,y_test,Y_train,Y_test = dset.setup_labels(labels,ratio,return_masks=True)

gs = gridspec.GridSpec(2,10)
fig = plt.figure(figsize=(10,4))
for i in range(10):
    ax = fig.add_subplot(gs[0,i])
    ax.imshow(X_train[i*40+500])
    ax = fig.add_subplot(gs[1,i])
    ax.imshow(Y_train[i*40+500])
    print(np.amax(Y_train[i*40+500]),y_train[i*40+500])

plt.show()

print(np.sum(y_train,axis=0)/y_train.shape[0])

model = models.new_cnn(inshape)
model.summary()
#train the model
print(inshape,X_train[0].shape)
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=nepochs)
model.save(f'{model_name}.hdf5')

plt.figure(figsize=(12,8))
plt.plot(np.arange(0, nepochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, nepochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, nepochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, nepochs), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

plt.savefig('./plots/history_new.png')
