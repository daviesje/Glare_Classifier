### Basic CNN for image classification of glare in overhead drone photography

To train a new net, use `python new_net.py [training image] [labels] [model name]`

To train an existing net with more data, use `python train_net.py [training image] [labels] [model name]`

both of these will output images of the training history and comparison plot

To use a net on new data, use `python predict_glare.py [image] [model name] [output suffix]`

this will output a comparison plot, and a .npy array of the classified glare

----

the other python files are modules that are imported in the above three scripts

----

The images you feed in should be smaller subsets of the larger image (4096x4096 is a decent size)

some example models are in the models/ directory
