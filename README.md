# MusicClassifier
Classifies music into genres using a neural net. The implementation in C++ is entirely written by me as an excercise in understanding how ANNs work under the hood. The NN uses backpropagation with gradient descent. A later addition into the NN was the use of momentum(inertia) and learning rate adaptation which reduced the training time for this particular dataset by about 20% while preserving the accuracy. A good explanation of both momentum and learning rate adaptation can be seen [here](https://www.willamette.edu/~gorr/classes/cs449/momrate.html).

The training dataset is taken from http://neuroph.sourceforge.net 

