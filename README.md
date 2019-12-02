### Basic CNN for image classification of glare in overhead drone photography

To train a new net, use `python new_net.py [training image] [labels] [model name]`

To train an existing net with more data, use `python train_net.py [training image] [labels] [model name]`

both of these will output images of the training history and comparison plot

To use a net on new data, use `python predict_glare.py [image] [model name] [output suffix]`

this will output a comparison plot, and a .npy array of the classified glare
