
# Image Classification
In this project, we'll classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).  
The dataset consists of airplanes, dogs, cats, and other objects.  
We'll preprocess the images, then train a convolutional neural network on all the samples.  
The images need to be normalized and the labels need to be one-hot encoded.  
We'll build a convolutional, max pooling, dropout, and fully connected layers.  
At the end, we'll get to see the neural network's predictions on the sample images.  

## Get the Data
Run the following cell to download the [CIFAR-10 dataset for python](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import problem_unittests as tests
import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'

# Use Floyd's cifar-10 dataset if present
floyd_cifar10_location = '/input/cifar-10/python.tar.gz'
if isfile(floyd_cifar10_location):
    tar_gz_path = floyd_cifar10_location
else:
    tar_gz_path = 'cifar-10-python.tar.gz'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile(tar_gz_path):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            tar_gz_path,
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open(tar_gz_path) as tar:
        tar.extractall()
        tar.close()


tests.test_folder_path(cifar10_dataset_folder_path)
```

    All files found!


## Explore the Data
The dataset is broken into batches to prevent a computer from running out of memory.  
The CIFAR-10 dataset consists of 5 batches, named `data_batch_1`, `data_batch_2`, etc.  
Each batch contains the labels and images that are one of the following:
* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

Understanding a dataset is part of making predictions on the data.  
Playing around with the code cell below by changing the `batch_id` and `sample_id`. The `batch_id` is the id for a batch (1-5).  
The `sample_id` is the id for a image and label pair in the batch.

Good questions to ask are "What are all possible labels?", "What is the range of values for the image data?", "Are the labels in order or random?".  
Answers to questions like these will help to preprocess the data and end up with better predictions.


```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import helper
import numpy as np
import sklearn


# Explore the dataset
batch_id = 1
sample_id = 5
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)
```

    
    Stats of batch 1:
    Samples: 10000
    Label Counts: {0: 1005, 1: 974, 2: 1032, 3: 1016, 4: 999, 5: 937, 6: 1030, 7: 1001, 8: 1025, 9: 981}
    First 20 Labels: [6, 9, 9, 4, 1, 1, 2, 7, 8, 3, 4, 7, 7, 2, 9, 9, 9, 3, 2, 6]
    
    Example of Image 5:
    Image - Min Value: 0 Max Value: 252
    Image - Shape: (32, 32, 3)
    Label - Label Id: 1 Name: automobile



![png](output_3_1.png)


## Preprocess Functions
### Normalize
The `normalize` function takes in image data, `x`, and returns it as a normalized Numpy array.  
The values are in the range of 0 to 1, inclusive.  
The return object is of the same shape as `x`.


```python
def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_normalize(normalize)
```

    Tests Passed


### One-hot encode
Another function for preprocessing.  
This time the `one_hot_encode` function.  
The input, `x`, is a list of labels.  
The function to returns the list of labels as One-Hot encoded Numpy array.  
The possible values for labels are 0 to 9.  
The one-hot encoding function returns the same encoding for each value between each call to `one_hot_encode`.  
The map of encodings is saved outside the function.


```python
lb_encoding = None
def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    global lb_encoding

    if lb_encoding is not None:
        return lb_encoding.transform(x)
    else:
        labels = np.array(x)
        lb_encoding = sklearn.preprocessing.LabelBinarizer()
        lb_encoding.fit(labels)
        return lb_encoding.transform(labels)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_one_hot_encode(one_hot_encode)
```

    Tests Passed


### Randomize Data
As we saw from exploring the data above, the order of the samples are randomized.  It doesn't hurt to randomize it again, but we don't need to for this dataset.

## Preprocess all the data and save it
Running the code cell below will preprocess all the CIFAR-10 data and save it to file. The code below also uses 10% of the training data for validation.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)
```

# Check Point
This is the first checkpoint.  If we ever decide to come back to this notebook or have to restart the notebook, we can start from here.  The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import pickle
import problem_unittests as tests
import helper

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
```

## Build the network
For the neural network, you'll build each layer into a function.  Most of the code you've seen has been outside of functions. To test your code more thoroughly, we require that you put each layer in a function.  This allows us to give you better feedback and test for simple mistakes using our unittests before you submit your project.

>**Note:** If you're finding it hard to dedicate enough time for this course each week, we've provided a small shortcut to this part of the project. In the next couple of problems, you'll have the option to use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages to build each layer, except the layers you build in the "Convolutional and Max Pooling Layer" section.  TF Layers is similar to Keras's and TFLearn's abstraction to layers, so it's easy to pickup.

>However, if you would like to get the most out of this course, try to solve all the problems _without_ using anything from the TF Layers packages. You **can** still use classes from other packages that happen to have the same name as ones you find in TF Layers! For example, instead of using the TF Layers version of the `conv2d` class, [tf.layers.conv2d](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d), you would want to use the TF Neural Network version of `conv2d`, [tf.nn.conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d). 

Let's begin!

### Input
The neural network needs to read the image data, one-hot encoded labels, and dropout keep probability. Implement the following functions
* Implement `neural_net_image_input`
 * Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)
 * Set the shape using `image_shape` with batch size set to `None`.
 * Name the TensorFlow placeholder "x" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).
* Implement `neural_net_label_input`
 * Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)
 * Set the shape using `n_classes` with batch size set to `None`.
 * Name the TensorFlow placeholder "y" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).
* Implement `neural_net_keep_prob_input`
 * Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) for dropout keep probability.
 * Name the TensorFlow placeholder "keep_prob" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).

These names will be used at the end of the project to load your saved model.

Note: `None` for shapes in TensorFlow allow for a dynamic size.


```python
import tensorflow as tf


def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    return tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], image_shape[2]], name='x')


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    return tf.placeholder(tf.float32, shape=[None, n_classes], name='y')


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32, None, name='keep_prob')


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)

```

    Image Input Tests Passed.
    Label Input Tests Passed.
    Keep Prob Tests Passed.


### Convolution and Max Pooling Layer
Convolution layers have a lot of success with images. For this code cell, you should implement the function `conv2d_maxpool` to apply convolution then max pooling:
* Create the weight and bias using `conv_ksize`, `conv_num_outputs` and the shape of `x_tensor`.
* Apply a convolution to `x_tensor` using weight and `conv_strides`.
 * We recommend you use same padding, but you're welcome to use any padding.
* Add bias
* Add a nonlinear activation to the convolution.
* Apply Max Pooling using `pool_ksize` and `pool_strides`.
 * We recommend you use same padding, but you're welcome to use any padding.

**Note:** You **can't** use [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) for **this** layer, but you can still use TensorFlow's [Neural Network](https://www.tensorflow.org/api_docs/python/tf/nn) package. You may still use the shortcut option for all the **other** layers.


```python
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor

    """
    weights = tf.Variable(tf.truncated_normal([conv_ksize[0],conv_ksize[1], x_tensor.get_shape().as_list()[3], conv_num_outputs]))
    bias = tf.Variable(tf.zeros(conv_num_outputs))
    
    convolution = tf.nn.conv2d(x_tensor, weights, strides=[1,conv_strides[0],conv_strides[1],1], padding='SAME')
    convolution_and_bias = tf.nn.bias_add(convolution, bias)
    
    convolution_and_bias_activation = tf.nn.relu(convolution_and_bias)
    
    max_pool = tf.nn.max_pool(convolution_and_bias_activation, [1,pool_ksize[0],pool_ksize[1],1], strides=[1,pool_strides[0],pool_strides[1],1], padding='SAME')
    return max_pool


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_con_pool(conv2d_maxpool)
```

    Tests Passed


### Flatten Layer
Implement the `flatten` function to change the dimension of `x_tensor` from a 4-D tensor to a 2-D tensor.  The output should be the shape (*Batch Size*, *Flattened Image Size*). Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.


```python
def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    return tf.contrib.layers.flatten(x_tensor)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_flatten(flatten)
```

    Tests Passed


### Fully-Connected Layer
Implement the `fully_conn` function to apply a fully connected layer to `x_tensor` with the shape (*Batch Size*, *num_outputs*). Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.


```python
def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    return tf.contrib.layers.fully_connected(x_tensor, num_outputs, activation_fn=tf.nn.relu, biases_initializer=tf.zeros_initializer(), trainable=True)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_fully_conn(fully_conn)
```

    Tests Passed


### Output Layer
Implement the `output` function to apply a fully connected layer to `x_tensor` with the shape (*Batch Size*, *num_outputs*). Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.

**Note:** Activation, softmax, or cross entropy should **not** be applied to this.


```python
def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    dimension = x_tensor.get_shape().as_list()
    shape = list( (dimension[-1],) + (num_outputs,))
    weight = tf.Variable(tf.truncated_normal(shape,0,0.01))
    bias = tf.Variable(tf.zeros(num_outputs))
    return tf.add(tf.matmul(x_tensor,weight), bias)
    # Implementation just using TensorFlow is commented below
    #return tf.contrib.layers.fully_connected(x_tensor, num_outputs, activation_fn=None)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_output(output)
```

    Tests Passed


### Create Convolutional Model
Implement the function `conv_net` to create a convolutional neural network model. The function takes in a batch of images, `x`, and outputs logits.  Use the layers you created above to create this model:

* Apply 1, 2, or 3 Convolution and Max Pool layers
* Apply a Flatten Layer
* Apply 1, 2, or 3 Fully Connected Layers
* Apply an Output Layer
* Return the output
* Apply [TensorFlow's Dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout) to one or more layers in the model using `keep_prob`. 


```python
def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    layer = conv2d_maxpool(x, 64, (4,4), (1,1), (2,2), (2,2))
    tf.nn.dropout(layer, keep_prob=keep_prob)
    
    # Apply a Flatten Layer
    layer = flatten(layer)

    # Apply 2 Fully Connected Layers
    layer = fully_conn(layer,500)
    layer = tf.nn.dropout(layer, keep_prob)
    layer = fully_conn(layer,100)
    layer = tf.nn.dropout(layer, keep_prob)
    
    # Return output
    return output(layer,10)
  
    

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)
```

    Neural Network Built!


## Train the Neural Network
### Single Optimization
Implement the function `train_neural_network` to do a single optimization.  The optimization should use `optimizer` to optimize in `session` with a `feed_dict` of the following:
* `x` for image input
* `y` for labels
* `keep_prob` for keep probability for dropout

This function will be called for each batch, so `tf.global_variables_initializer()` has already been called.

Note: Nothing needs to be returned. This function is only optimizing the neural network.


```python
def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # TODO: Implement Function
    session.run(optimizer, feed_dict={x:feature_batch, y:label_batch, keep_prob:keep_probability})
    

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_train_nn(train_neural_network)
```

    Tests Passed


### Show Stats
Implement the function `print_stats` to print loss and validation accuracy.  Use the global variables `valid_features` and `valid_labels` to calculate validation accuracy.  Use a keep probability of `1.0` to calculate the loss and validation accuracy.


```python
def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    loss = session.run(cost, feed_dict={x: feature_batch, y:label_batch, keep_prob:1.})
    valid_accuracy = session.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.})
    
    print('Loss: {:>10.4f}'.format(loss))
    print('Accuracy: {:.6f}'.format(valid_accuracy))
```

### Hyperparameters
Tune the following parameters:
* Set `epochs` to the number of iterations until the network stops learning or start overfitting
* Set `batch_size` to the highest number that your machine has memory for.  Most people set them to common sizes of memory:
 * 64
 * 128
 * 256
 * ...
* Set `keep_probability` to the probability of keeping a node using dropout


```python
# TODO: Tune Parameters
epochs = 100
batch_size = 512
keep_probability = 0.5
```

### Train on a Single CIFAR-10 Batch
Instead of training the neural network on all the CIFAR-10 batches of data, let's use a single batch. This should save time while you iterate on the model to get a better accuracy.  Once the final validation accuracy is 50% or greater, run the model on all the data in the next section.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)
```

    Checking the Training on a Single Batch...
    Epoch  1, CIFAR-10 Batch 1:  Loss:     2.2741
    Accuracy: 0.159200
    Epoch  2, CIFAR-10 Batch 1:  Loss:     2.0728
    Accuracy: 0.240600
    Epoch  3, CIFAR-10 Batch 1:  Loss:     1.9645
    Accuracy: 0.285800
    Epoch  4, CIFAR-10 Batch 1:  Loss:     1.8730
    Accuracy: 0.336400
    Epoch  5, CIFAR-10 Batch 1:  Loss:     1.7617
    Accuracy: 0.372600
    Epoch  6, CIFAR-10 Batch 1:  Loss:     1.7057
    Accuracy: 0.392000
    Epoch  7, CIFAR-10 Batch 1:  Loss:     1.6158
    Accuracy: 0.392800
    Epoch  8, CIFAR-10 Batch 1:  Loss:     1.5579
    Accuracy: 0.429800
    Epoch  9, CIFAR-10 Batch 1:  Loss:     1.4939
    Accuracy: 0.424200
    Epoch 10, CIFAR-10 Batch 1:  Loss:     1.4390
    Accuracy: 0.444000
    Epoch 11, CIFAR-10 Batch 1:  Loss:     1.4059
    Accuracy: 0.452400
    Epoch 12, CIFAR-10 Batch 1:  Loss:     1.3298
    Accuracy: 0.464800
    Epoch 13, CIFAR-10 Batch 1:  Loss:     1.3332
    Accuracy: 0.450200
    Epoch 14, CIFAR-10 Batch 1:  Loss:     1.2317
    Accuracy: 0.488400
    Epoch 15, CIFAR-10 Batch 1:  Loss:     1.1988
    Accuracy: 0.500000
    Epoch 16, CIFAR-10 Batch 1:  Loss:     1.1245
    Accuracy: 0.490800
    Epoch 17, CIFAR-10 Batch 1:  Loss:     1.0743
    Accuracy: 0.502800
    Epoch 18, CIFAR-10 Batch 1:  Loss:     1.0410
    Accuracy: 0.503200
    Epoch 19, CIFAR-10 Batch 1:  Loss:     0.9986
    Accuracy: 0.516200
    Epoch 20, CIFAR-10 Batch 1:  Loss:     0.9680
    Accuracy: 0.502800
    Epoch 21, CIFAR-10 Batch 1:  Loss:     0.9311
    Accuracy: 0.517200
    Epoch 22, CIFAR-10 Batch 1:  Loss:     0.8723
    Accuracy: 0.529200
    Epoch 23, CIFAR-10 Batch 1:  Loss:     0.8511
    Accuracy: 0.526600
    Epoch 24, CIFAR-10 Batch 1:  Loss:     0.8240
    Accuracy: 0.518200
    Epoch 25, CIFAR-10 Batch 1:  Loss:     0.7855
    Accuracy: 0.529200
    Epoch 26, CIFAR-10 Batch 1:  Loss:     0.8013
    Accuracy: 0.522600
    Epoch 27, CIFAR-10 Batch 1:  Loss:     0.7687
    Accuracy: 0.520400
    Epoch 28, CIFAR-10 Batch 1:  Loss:     0.7500
    Accuracy: 0.518600
    Epoch 29, CIFAR-10 Batch 1:  Loss:     0.7193
    Accuracy: 0.522800
    Epoch 30, CIFAR-10 Batch 1:  Loss:     0.6725
    Accuracy: 0.523400
    Epoch 31, CIFAR-10 Batch 1:  Loss:     0.6301
    Accuracy: 0.526200
    Epoch 32, CIFAR-10 Batch 1:  Loss:     0.5753
    Accuracy: 0.541600
    Epoch 33, CIFAR-10 Batch 1:  Loss:     0.5785
    Accuracy: 0.530400
    Epoch 34, CIFAR-10 Batch 1:  Loss:     0.5513
    Accuracy: 0.534200
    Epoch 35, CIFAR-10 Batch 1:  Loss:     0.5124
    Accuracy: 0.534000
    Epoch 36, CIFAR-10 Batch 1:  Loss:     0.5439
    Accuracy: 0.537200
    Epoch 37, CIFAR-10 Batch 1:  Loss:     0.5174
    Accuracy: 0.546000
    Epoch 38, CIFAR-10 Batch 1:  Loss:     0.4844
    Accuracy: 0.537000
    Epoch 39, CIFAR-10 Batch 1:  Loss:     0.4504
    Accuracy: 0.540200
    Epoch 40, CIFAR-10 Batch 1:  Loss:     0.4402
    Accuracy: 0.530600
    Epoch 41, CIFAR-10 Batch 1:  Loss:     0.4242
    Accuracy: 0.531000
    Epoch 42, CIFAR-10 Batch 1:  Loss:     0.4097
    Accuracy: 0.541200
    Epoch 43, CIFAR-10 Batch 1:  Loss:     0.4430
    Accuracy: 0.536200
    Epoch 44, CIFAR-10 Batch 1:  Loss:     0.3798
    Accuracy: 0.541600
    Epoch 45, CIFAR-10 Batch 1:  Loss:     0.3413
    Accuracy: 0.545800
    Epoch 46, CIFAR-10 Batch 1:  Loss:     0.3427
    Accuracy: 0.544200
    Epoch 47, CIFAR-10 Batch 1:  Loss:     0.3457
    Accuracy: 0.552000
    Epoch 48, CIFAR-10 Batch 1:  Loss:     0.3256
    Accuracy: 0.539000
    Epoch 49, CIFAR-10 Batch 1:  Loss:     0.3233
    Accuracy: 0.550600
    Epoch 50, CIFAR-10 Batch 1:  Loss:     0.2978
    Accuracy: 0.550800
    Epoch 51, CIFAR-10 Batch 1:  Loss:     0.3245
    Accuracy: 0.538800
    Epoch 52, CIFAR-10 Batch 1:  Loss:     0.3052
    Accuracy: 0.547800
    Epoch 53, CIFAR-10 Batch 1:  Loss:     0.2695
    Accuracy: 0.549200
    Epoch 54, CIFAR-10 Batch 1:  Loss:     0.2550
    Accuracy: 0.544600
    Epoch 55, CIFAR-10 Batch 1:  Loss:     0.2730
    Accuracy: 0.544600
    Epoch 56, CIFAR-10 Batch 1:  Loss:     0.2652
    Accuracy: 0.552600
    Epoch 57, CIFAR-10 Batch 1:  Loss:     0.2726
    Accuracy: 0.539800
    Epoch 58, CIFAR-10 Batch 1:  Loss:     0.2360
    Accuracy: 0.540200
    Epoch 59, CIFAR-10 Batch 1:  Loss:     0.2193
    Accuracy: 0.552600
    Epoch 60, CIFAR-10 Batch 1:  Loss:     0.2399
    Accuracy: 0.554600
    Epoch 61, CIFAR-10 Batch 1:  Loss:     0.2418
    Accuracy: 0.544600
    Epoch 62, CIFAR-10 Batch 1:  Loss:     0.2363
    Accuracy: 0.543600
    Epoch 63, CIFAR-10 Batch 1:  Loss:     0.2043
    Accuracy: 0.545600
    Epoch 64, CIFAR-10 Batch 1:  Loss:     0.2109
    Accuracy: 0.538800
    Epoch 65, CIFAR-10 Batch 1:  Loss:     0.2069
    Accuracy: 0.543400
    Epoch 66, CIFAR-10 Batch 1:  Loss:     0.2025
    Accuracy: 0.549200
    Epoch 67, CIFAR-10 Batch 1:  Loss:     0.1713
    Accuracy: 0.550800
    Epoch 68, CIFAR-10 Batch 1:  Loss:     0.1739
    Accuracy: 0.554800
    Epoch 69, CIFAR-10 Batch 1:  Loss:     0.1604
    Accuracy: 0.554400
    Epoch 70, CIFAR-10 Batch 1:  Loss:     0.1651
    Accuracy: 0.559600
    Epoch 71, CIFAR-10 Batch 1:  Loss:     0.1621
    Accuracy: 0.552400
    Epoch 72, CIFAR-10 Batch 1:  Loss:     0.1432
    Accuracy: 0.558000
    Epoch 73, CIFAR-10 Batch 1:  Loss:     0.1350
    Accuracy: 0.563400
    Epoch 74, CIFAR-10 Batch 1:  Loss:     0.1361
    Accuracy: 0.550600
    Epoch 75, CIFAR-10 Batch 1:  Loss:     0.1241
    Accuracy: 0.558600
    Epoch 76, CIFAR-10 Batch 1:  Loss:     0.1239
    Accuracy: 0.556800
    Epoch 77, CIFAR-10 Batch 1:  Loss:     0.1249
    Accuracy: 0.549800
    Epoch 78, CIFAR-10 Batch 1:  Loss:     0.1183
    Accuracy: 0.555400
    Epoch 79, CIFAR-10 Batch 1:  Loss:     0.1109
    Accuracy: 0.554600
    Epoch 80, CIFAR-10 Batch 1:  Loss:     0.0996
    Accuracy: 0.546600
    Epoch 81, CIFAR-10 Batch 1:  Loss:     0.1082
    Accuracy: 0.552200
    Epoch 82, CIFAR-10 Batch 1:  Loss:     0.0896
    Accuracy: 0.544600
    Epoch 83, CIFAR-10 Batch 1:  Loss:     0.0927
    Accuracy: 0.558400
    Epoch 84, CIFAR-10 Batch 1:  Loss:     0.0891
    Accuracy: 0.551800
    Epoch 85, CIFAR-10 Batch 1:  Loss:     0.0856
    Accuracy: 0.554000
    Epoch 86, CIFAR-10 Batch 1:  Loss:     0.0794
    Accuracy: 0.559000
    Epoch 87, CIFAR-10 Batch 1:  Loss:     0.0770
    Accuracy: 0.558600
    Epoch 88, CIFAR-10 Batch 1:  Loss:     0.0765
    Accuracy: 0.561000
    Epoch 89, CIFAR-10 Batch 1:  Loss:     0.0801
    Accuracy: 0.562400
    Epoch 90, CIFAR-10 Batch 1:  Loss:     0.0768
    Accuracy: 0.556800
    Epoch 91, CIFAR-10 Batch 1:  Loss:     0.0765
    Accuracy: 0.561200
    Epoch 92, CIFAR-10 Batch 1:  Loss:     0.0587
    Accuracy: 0.556200
    Epoch 93, CIFAR-10 Batch 1:  Loss:     0.0767
    Accuracy: 0.557800
    Epoch 94, CIFAR-10 Batch 1:  Loss:     0.0717
    Accuracy: 0.556800
    Epoch 95, CIFAR-10 Batch 1:  Loss:     0.0728
    Accuracy: 0.560400
    Epoch 96, CIFAR-10 Batch 1:  Loss:     0.0674
    Accuracy: 0.562800
    Epoch 97, CIFAR-10 Batch 1:  Loss:     0.0756
    Accuracy: 0.555800
    Epoch 98, CIFAR-10 Batch 1:  Loss:     0.0717
    Accuracy: 0.557600
    Epoch 99, CIFAR-10 Batch 1:  Loss:     0.0767
    Accuracy: 0.563800
    Epoch 100, CIFAR-10 Batch 1:  Loss:     0.0674
    Accuracy: 0.554400


### Fully Train the Model
Now that you got a good accuracy with a single CIFAR-10 batch, try it with all five batches.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
            
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
```

    Training...
    Epoch  1, CIFAR-10 Batch 1:  Loss:     2.2740
    Accuracy: 0.191400
    Epoch  1, CIFAR-10 Batch 2:  Loss:     2.0493
    Accuracy: 0.257600
    Epoch  1, CIFAR-10 Batch 3:  Loss:     1.8533
    Accuracy: 0.305800
    Epoch  1, CIFAR-10 Batch 4:  Loss:     1.7334
    Accuracy: 0.370800
    Epoch  1, CIFAR-10 Batch 5:  Loss:     1.7248
    Accuracy: 0.401000
    Epoch  2, CIFAR-10 Batch 1:  Loss:     1.7207
    Accuracy: 0.393600
    Epoch  2, CIFAR-10 Batch 2:  Loss:     1.5546
    Accuracy: 0.419000
    Epoch  2, CIFAR-10 Batch 3:  Loss:     1.4243
    Accuracy: 0.430400
    Epoch  2, CIFAR-10 Batch 4:  Loss:     1.4280
    Accuracy: 0.443000
    Epoch  2, CIFAR-10 Batch 5:  Loss:     1.4505
    Accuracy: 0.467400
    Epoch  3, CIFAR-10 Batch 1:  Loss:     1.5237
    Accuracy: 0.466400
    Epoch  3, CIFAR-10 Batch 2:  Loss:     1.3811
    Accuracy: 0.473600
    Epoch  3, CIFAR-10 Batch 3:  Loss:     1.2799
    Accuracy: 0.467000
    Epoch  3, CIFAR-10 Batch 4:  Loss:     1.2639
    Accuracy: 0.485400
    Epoch  3, CIFAR-10 Batch 5:  Loss:     1.3058
    Accuracy: 0.498800
    Epoch  4, CIFAR-10 Batch 1:  Loss:     1.4044
    Accuracy: 0.494200
    Epoch  4, CIFAR-10 Batch 2:  Loss:     1.2766
    Accuracy: 0.507200
    Epoch  4, CIFAR-10 Batch 3:  Loss:     1.1631
    Accuracy: 0.504800
    Epoch  4, CIFAR-10 Batch 4:  Loss:     1.1875
    Accuracy: 0.513600
    Epoch  4, CIFAR-10 Batch 5:  Loss:     1.1926
    Accuracy: 0.532800
    Epoch  5, CIFAR-10 Batch 1:  Loss:     1.2536
    Accuracy: 0.539800
    Epoch  5, CIFAR-10 Batch 2:  Loss:     1.1533
    Accuracy: 0.538600
    Epoch  5, CIFAR-10 Batch 3:  Loss:     1.0515
    Accuracy: 0.539200
    Epoch  5, CIFAR-10 Batch 4:  Loss:     1.0981
    Accuracy: 0.548000
    Epoch  5, CIFAR-10 Batch 5:  Loss:     1.0772
    Accuracy: 0.556400
    Epoch  6, CIFAR-10 Batch 1:  Loss:     1.1876
    Accuracy: 0.548800
    Epoch  6, CIFAR-10 Batch 2:  Loss:     1.0804
    Accuracy: 0.561200
    Epoch  6, CIFAR-10 Batch 3:  Loss:     0.9763
    Accuracy: 0.565600
    Epoch  6, CIFAR-10 Batch 4:  Loss:     1.0355
    Accuracy: 0.568600
    Epoch  6, CIFAR-10 Batch 5:  Loss:     0.9942
    Accuracy: 0.566800
    Epoch  7, CIFAR-10 Batch 1:  Loss:     1.0854
    Accuracy: 0.572800
    Epoch  7, CIFAR-10 Batch 2:  Loss:     1.0064
    Accuracy: 0.576600
    Epoch  7, CIFAR-10 Batch 3:  Loss:     0.9067
    Accuracy: 0.569600
    Epoch  7, CIFAR-10 Batch 4:  Loss:     0.9718
    Accuracy: 0.580000
    Epoch  7, CIFAR-10 Batch 5:  Loss:     0.9460
    Accuracy: 0.580800
    Epoch  8, CIFAR-10 Batch 1:  Loss:     1.0132
    Accuracy: 0.586400
    Epoch  8, CIFAR-10 Batch 2:  Loss:     0.9411
    Accuracy: 0.585600
    Epoch  8, CIFAR-10 Batch 3:  Loss:     0.8423
    Accuracy: 0.583600
    Epoch  8, CIFAR-10 Batch 4:  Loss:     0.9069
    Accuracy: 0.581000
    Epoch  8, CIFAR-10 Batch 5:  Loss:     0.8663
    Accuracy: 0.588400
    Epoch  9, CIFAR-10 Batch 1:  Loss:     0.9759
    Accuracy: 0.584800
    Epoch  9, CIFAR-10 Batch 2:  Loss:     0.8943
    Accuracy: 0.587800
    Epoch  9, CIFAR-10 Batch 3:  Loss:     0.8552
    Accuracy: 0.585200
    Epoch  9, CIFAR-10 Batch 4:  Loss:     0.8568
    Accuracy: 0.580600
    Epoch  9, CIFAR-10 Batch 5:  Loss:     0.8175
    Accuracy: 0.603400
    Epoch 10, CIFAR-10 Batch 1:  Loss:     0.9361
    Accuracy: 0.591000
    Epoch 10, CIFAR-10 Batch 2:  Loss:     0.8515
    Accuracy: 0.597200
    Epoch 10, CIFAR-10 Batch 3:  Loss:     0.8041
    Accuracy: 0.595400
    Epoch 10, CIFAR-10 Batch 4:  Loss:     0.8099
    Accuracy: 0.599000
    Epoch 10, CIFAR-10 Batch 5:  Loss:     0.7897
    Accuracy: 0.605400
    Epoch 11, CIFAR-10 Batch 1:  Loss:     0.8804
    Accuracy: 0.601800
    Epoch 11, CIFAR-10 Batch 2:  Loss:     0.7989
    Accuracy: 0.604000
    Epoch 11, CIFAR-10 Batch 3:  Loss:     0.7587
    Accuracy: 0.596200
    Epoch 11, CIFAR-10 Batch 4:  Loss:     0.7657
    Accuracy: 0.605200
    Epoch 11, CIFAR-10 Batch 5:  Loss:     0.7342
    Accuracy: 0.606200
    Epoch 12, CIFAR-10 Batch 1:  Loss:     0.8349
    Accuracy: 0.609200
    Epoch 12, CIFAR-10 Batch 2:  Loss:     0.7855
    Accuracy: 0.605200
    Epoch 12, CIFAR-10 Batch 3:  Loss:     0.7085
    Accuracy: 0.612600
    Epoch 12, CIFAR-10 Batch 4:  Loss:     0.7351
    Accuracy: 0.612400
    Epoch 12, CIFAR-10 Batch 5:  Loss:     0.7091
    Accuracy: 0.605400
    Epoch 13, CIFAR-10 Batch 1:  Loss:     0.8235
    Accuracy: 0.606200
    Epoch 13, CIFAR-10 Batch 2:  Loss:     0.7455
    Accuracy: 0.612200
    Epoch 13, CIFAR-10 Batch 3:  Loss:     0.6876
    Accuracy: 0.609600
    Epoch 13, CIFAR-10 Batch 4:  Loss:     0.7130
    Accuracy: 0.617000
    Epoch 13, CIFAR-10 Batch 5:  Loss:     0.6753
    Accuracy: 0.620200
    Epoch 14, CIFAR-10 Batch 1:  Loss:     0.7752
    Accuracy: 0.615800
    Epoch 14, CIFAR-10 Batch 2:  Loss:     0.7367
    Accuracy: 0.611200
    Epoch 14, CIFAR-10 Batch 3:  Loss:     0.6341
    Accuracy: 0.617400
    Epoch 14, CIFAR-10 Batch 4:  Loss:     0.6571
    Accuracy: 0.619200
    Epoch 14, CIFAR-10 Batch 5:  Loss:     0.6387
    Accuracy: 0.618400
    Epoch 15, CIFAR-10 Batch 1:  Loss:     0.7081
    Accuracy: 0.618000
    Epoch 15, CIFAR-10 Batch 2:  Loss:     0.6640
    Accuracy: 0.618800
    Epoch 15, CIFAR-10 Batch 3:  Loss:     0.6042
    Accuracy: 0.620200
    Epoch 15, CIFAR-10 Batch 4:  Loss:     0.6425
    Accuracy: 0.623400
    Epoch 15, CIFAR-10 Batch 5:  Loss:     0.6315
    Accuracy: 0.615800
    Epoch 16, CIFAR-10 Batch 1:  Loss:     0.6888
    Accuracy: 0.622400
    Epoch 16, CIFAR-10 Batch 2:  Loss:     0.6402
    Accuracy: 0.629000
    Epoch 16, CIFAR-10 Batch 3:  Loss:     0.6064
    Accuracy: 0.622200
    Epoch 16, CIFAR-10 Batch 4:  Loss:     0.6199
    Accuracy: 0.628200
    Epoch 16, CIFAR-10 Batch 5:  Loss:     0.5833
    Accuracy: 0.627200
    Epoch 17, CIFAR-10 Batch 1:  Loss:     0.6609
    Accuracy: 0.628400
    Epoch 17, CIFAR-10 Batch 2:  Loss:     0.6440
    Accuracy: 0.623600
    Epoch 17, CIFAR-10 Batch 3:  Loss:     0.5946
    Accuracy: 0.618400
    Epoch 17, CIFAR-10 Batch 4:  Loss:     0.6051
    Accuracy: 0.625800
    Epoch 17, CIFAR-10 Batch 5:  Loss:     0.5744
    Accuracy: 0.623800
    Epoch 18, CIFAR-10 Batch 1:  Loss:     0.6245
    Accuracy: 0.625600
    Epoch 18, CIFAR-10 Batch 2:  Loss:     0.6122
    Accuracy: 0.629800
    Epoch 18, CIFAR-10 Batch 3:  Loss:     0.5631
    Accuracy: 0.627600
    Epoch 18, CIFAR-10 Batch 4:  Loss:     0.5838
    Accuracy: 0.624800
    Epoch 18, CIFAR-10 Batch 5:  Loss:     0.5398
    Accuracy: 0.626800
    Epoch 19, CIFAR-10 Batch 1:  Loss:     0.6303
    Accuracy: 0.626800
    Epoch 19, CIFAR-10 Batch 2:  Loss:     0.6007
    Accuracy: 0.629600
    Epoch 19, CIFAR-10 Batch 3:  Loss:     0.5293
    Accuracy: 0.629600
    Epoch 19, CIFAR-10 Batch 4:  Loss:     0.5686
    Accuracy: 0.627600
    Epoch 19, CIFAR-10 Batch 5:  Loss:     0.5145
    Accuracy: 0.631000
    Epoch 20, CIFAR-10 Batch 1:  Loss:     0.5765
    Accuracy: 0.633200
    Epoch 20, CIFAR-10 Batch 2:  Loss:     0.5736
    Accuracy: 0.631600
    Epoch 20, CIFAR-10 Batch 3:  Loss:     0.5330
    Accuracy: 0.624200
    Epoch 20, CIFAR-10 Batch 4:  Loss:     0.5316
    Accuracy: 0.635000
    Epoch 20, CIFAR-10 Batch 5:  Loss:     0.5134
    Accuracy: 0.627800
    Epoch 21, CIFAR-10 Batch 1:  Loss:     0.5657
    Accuracy: 0.634400
    Epoch 21, CIFAR-10 Batch 2:  Loss:     0.5481
    Accuracy: 0.632000
    Epoch 21, CIFAR-10 Batch 3:  Loss:     0.4640
    Accuracy: 0.640600
    Epoch 21, CIFAR-10 Batch 4:  Loss:     0.5179
    Accuracy: 0.627400
    Epoch 21, CIFAR-10 Batch 5:  Loss:     0.4900
    Accuracy: 0.628000
    Epoch 22, CIFAR-10 Batch 1:  Loss:     0.5310
    Accuracy: 0.635400
    Epoch 22, CIFAR-10 Batch 2:  Loss:     0.5021
    Accuracy: 0.639200
    Epoch 22, CIFAR-10 Batch 3:  Loss:     0.4728
    Accuracy: 0.633400
    Epoch 22, CIFAR-10 Batch 4:  Loss:     0.5182
    Accuracy: 0.631400
    Epoch 22, CIFAR-10 Batch 5:  Loss:     0.4858
    Accuracy: 0.627800
    Epoch 23, CIFAR-10 Batch 1:  Loss:     0.5141
    Accuracy: 0.632600
    Epoch 23, CIFAR-10 Batch 2:  Loss:     0.4982
    Accuracy: 0.629800
    Epoch 23, CIFAR-10 Batch 3:  Loss:     0.4405
    Accuracy: 0.635800
    Epoch 23, CIFAR-10 Batch 4:  Loss:     0.4856
    Accuracy: 0.629000
    Epoch 23, CIFAR-10 Batch 5:  Loss:     0.4512
    Accuracy: 0.631000
    Epoch 24, CIFAR-10 Batch 1:  Loss:     0.4858
    Accuracy: 0.639200
    Epoch 24, CIFAR-10 Batch 2:  Loss:     0.4674
    Accuracy: 0.640000
    Epoch 24, CIFAR-10 Batch 3:  Loss:     0.4449
    Accuracy: 0.638600
    Epoch 24, CIFAR-10 Batch 4:  Loss:     0.5062
    Accuracy: 0.630400
    Epoch 24, CIFAR-10 Batch 5:  Loss:     0.4316
    Accuracy: 0.638800
    Epoch 25, CIFAR-10 Batch 1:  Loss:     0.5093
    Accuracy: 0.633000
    Epoch 25, CIFAR-10 Batch 2:  Loss:     0.4663
    Accuracy: 0.639200
    Epoch 25, CIFAR-10 Batch 3:  Loss:     0.4213
    Accuracy: 0.633800
    Epoch 25, CIFAR-10 Batch 4:  Loss:     0.4414
    Accuracy: 0.640400
    Epoch 25, CIFAR-10 Batch 5:  Loss:     0.3919
    Accuracy: 0.639000
    Epoch 26, CIFAR-10 Batch 1:  Loss:     0.4604
    Accuracy: 0.633600
    Epoch 26, CIFAR-10 Batch 2:  Loss:     0.4478
    Accuracy: 0.639600
    Epoch 26, CIFAR-10 Batch 3:  Loss:     0.4152
    Accuracy: 0.630600
    Epoch 26, CIFAR-10 Batch 4:  Loss:     0.4331
    Accuracy: 0.632200
    Epoch 26, CIFAR-10 Batch 5:  Loss:     0.3908
    Accuracy: 0.636400
    Epoch 27, CIFAR-10 Batch 1:  Loss:     0.4586
    Accuracy: 0.638200
    Epoch 27, CIFAR-10 Batch 2:  Loss:     0.4317
    Accuracy: 0.635800
    Epoch 27, CIFAR-10 Batch 3:  Loss:     0.3911
    Accuracy: 0.642200
    Epoch 27, CIFAR-10 Batch 4:  Loss:     0.4453
    Accuracy: 0.639800
    Epoch 27, CIFAR-10 Batch 5:  Loss:     0.3741
    Accuracy: 0.645200
    Epoch 28, CIFAR-10 Batch 1:  Loss:     0.4130
    Accuracy: 0.642400
    Epoch 28, CIFAR-10 Batch 2:  Loss:     0.4123
    Accuracy: 0.639800
    Epoch 28, CIFAR-10 Batch 3:  Loss:     0.3659
    Accuracy: 0.640000
    Epoch 28, CIFAR-10 Batch 4:  Loss:     0.3900
    Accuracy: 0.638400
    Epoch 28, CIFAR-10 Batch 5:  Loss:     0.3771
    Accuracy: 0.637800
    Epoch 29, CIFAR-10 Batch 1:  Loss:     0.4148
    Accuracy: 0.637800
    Epoch 29, CIFAR-10 Batch 2:  Loss:     0.3832
    Accuracy: 0.640600
    Epoch 29, CIFAR-10 Batch 3:  Loss:     0.3420
    Accuracy: 0.638800
    Epoch 29, CIFAR-10 Batch 4:  Loss:     0.3820
    Accuracy: 0.638400
    Epoch 29, CIFAR-10 Batch 5:  Loss:     0.3723
    Accuracy: 0.639200
    Epoch 30, CIFAR-10 Batch 1:  Loss:     0.4162
    Accuracy: 0.640800
    Epoch 30, CIFAR-10 Batch 2:  Loss:     0.3965
    Accuracy: 0.639600
    Epoch 30, CIFAR-10 Batch 3:  Loss:     0.3554
    Accuracy: 0.633600
    Epoch 30, CIFAR-10 Batch 4:  Loss:     0.3849
    Accuracy: 0.636200
    Epoch 30, CIFAR-10 Batch 5:  Loss:     0.3422
    Accuracy: 0.636200
    Epoch 31, CIFAR-10 Batch 1:  Loss:     0.4156
    Accuracy: 0.637600
    Epoch 31, CIFAR-10 Batch 2:  Loss:     0.3760
    Accuracy: 0.641200
    Epoch 31, CIFAR-10 Batch 3:  Loss:     0.3256
    Accuracy: 0.639200
    Epoch 31, CIFAR-10 Batch 4:  Loss:     0.3667
    Accuracy: 0.643000
    Epoch 31, CIFAR-10 Batch 5:  Loss:     0.3383
    Accuracy: 0.649200
    Epoch 32, CIFAR-10 Batch 1:  Loss:     0.3898
    Accuracy: 0.640200
    Epoch 32, CIFAR-10 Batch 2:  Loss:     0.3838
    Accuracy: 0.639200
    Epoch 32, CIFAR-10 Batch 3:  Loss:     0.3237
    Accuracy: 0.629000
    Epoch 32, CIFAR-10 Batch 4:  Loss:     0.3764
    Accuracy: 0.633400
    Epoch 32, CIFAR-10 Batch 5:  Loss:     0.3289
    Accuracy: 0.644000
    Epoch 33, CIFAR-10 Batch 1:  Loss:     0.3508
    Accuracy: 0.647000
    Epoch 33, CIFAR-10 Batch 2:  Loss:     0.3750
    Accuracy: 0.645400
    Epoch 33, CIFAR-10 Batch 3:  Loss:     0.3186
    Accuracy: 0.637400
    Epoch 33, CIFAR-10 Batch 4:  Loss:     0.3607
    Accuracy: 0.636400
    Epoch 33, CIFAR-10 Batch 5:  Loss:     0.3161
    Accuracy: 0.653200
    Epoch 34, CIFAR-10 Batch 1:  Loss:     0.3506
    Accuracy: 0.647600
    Epoch 34, CIFAR-10 Batch 2:  Loss:     0.3162
    Accuracy: 0.646800
    Epoch 34, CIFAR-10 Batch 3:  Loss:     0.2843
    Accuracy: 0.639800
    Epoch 34, CIFAR-10 Batch 4:  Loss:     0.3444
    Accuracy: 0.648600
    Epoch 34, CIFAR-10 Batch 5:  Loss:     0.3132
    Accuracy: 0.641600
    Epoch 35, CIFAR-10 Batch 1:  Loss:     0.3579
    Accuracy: 0.644600
    Epoch 35, CIFAR-10 Batch 2:  Loss:     0.3347
    Accuracy: 0.643200
    Epoch 35, CIFAR-10 Batch 3:  Loss:     0.2736
    Accuracy: 0.644200
    Epoch 35, CIFAR-10 Batch 4:  Loss:     0.3252
    Accuracy: 0.643600
    Epoch 35, CIFAR-10 Batch 5:  Loss:     0.3019
    Accuracy: 0.643000
    Epoch 36, CIFAR-10 Batch 1:  Loss:     0.3370
    Accuracy: 0.642400
    Epoch 36, CIFAR-10 Batch 2:  Loss:     0.3228
    Accuracy: 0.648200
    Epoch 36, CIFAR-10 Batch 3:  Loss:     0.2758
    Accuracy: 0.643600
    Epoch 36, CIFAR-10 Batch 4:  Loss:     0.3151
    Accuracy: 0.640400
    Epoch 36, CIFAR-10 Batch 5:  Loss:     0.2788
    Accuracy: 0.650400
    Epoch 37, CIFAR-10 Batch 1:  Loss:     0.3332
    Accuracy: 0.648400
    Epoch 37, CIFAR-10 Batch 2:  Loss:     0.3011
    Accuracy: 0.645800
    Epoch 37, CIFAR-10 Batch 3:  Loss:     0.2574
    Accuracy: 0.646000
    Epoch 37, CIFAR-10 Batch 4:  Loss:     0.3147
    Accuracy: 0.636000
    Epoch 37, CIFAR-10 Batch 5:  Loss:     0.2663
    Accuracy: 0.646400
    Epoch 38, CIFAR-10 Batch 1:  Loss:     0.3134
    Accuracy: 0.646800
    Epoch 38, CIFAR-10 Batch 2:  Loss:     0.2965
    Accuracy: 0.642400
    Epoch 38, CIFAR-10 Batch 3:  Loss:     0.2557
    Accuracy: 0.648200
    Epoch 38, CIFAR-10 Batch 4:  Loss:     0.2867
    Accuracy: 0.648000
    Epoch 38, CIFAR-10 Batch 5:  Loss:     0.2640
    Accuracy: 0.649000
    Epoch 39, CIFAR-10 Batch 1:  Loss:     0.2981
    Accuracy: 0.645800
    Epoch 39, CIFAR-10 Batch 2:  Loss:     0.2709
    Accuracy: 0.644200
    Epoch 39, CIFAR-10 Batch 3:  Loss:     0.2555
    Accuracy: 0.639200
    Epoch 39, CIFAR-10 Batch 4:  Loss:     0.2893
    Accuracy: 0.634800
    Epoch 39, CIFAR-10 Batch 5:  Loss:     0.2596
    Accuracy: 0.648000
    Epoch 40, CIFAR-10 Batch 1:  Loss:     0.3064
    Accuracy: 0.650400
    Epoch 40, CIFAR-10 Batch 2:  Loss:     0.3002
    Accuracy: 0.649800
    Epoch 40, CIFAR-10 Batch 3:  Loss:     0.2426
    Accuracy: 0.647000
    Epoch 40, CIFAR-10 Batch 4:  Loss:     0.2889
    Accuracy: 0.639200
    Epoch 40, CIFAR-10 Batch 5:  Loss:     0.2448
    Accuracy: 0.638400
    Epoch 41, CIFAR-10 Batch 1:  Loss:     0.2979
    Accuracy: 0.641000
    Epoch 41, CIFAR-10 Batch 2:  Loss:     0.2779
    Accuracy: 0.642800
    Epoch 41, CIFAR-10 Batch 3:  Loss:     0.2511
    Accuracy: 0.643200
    Epoch 41, CIFAR-10 Batch 4:  Loss:     0.2876
    Accuracy: 0.646800
    Epoch 41, CIFAR-10 Batch 5:  Loss:     0.2440
    Accuracy: 0.645000
    Epoch 42, CIFAR-10 Batch 1:  Loss:     0.2912
    Accuracy: 0.643200
    Epoch 42, CIFAR-10 Batch 2:  Loss:     0.2757
    Accuracy: 0.642600
    Epoch 42, CIFAR-10 Batch 3:  Loss:     0.2373
    Accuracy: 0.637400
    Epoch 42, CIFAR-10 Batch 4:  Loss:     0.2728
    Accuracy: 0.642400
    Epoch 42, CIFAR-10 Batch 5:  Loss:     0.2491
    Accuracy: 0.643600
    Epoch 43, CIFAR-10 Batch 1:  Loss:     0.2857
    Accuracy: 0.641800
    Epoch 43, CIFAR-10 Batch 2:  Loss:     0.2747
    Accuracy: 0.645400
    Epoch 43, CIFAR-10 Batch 3:  Loss:     0.2328
    Accuracy: 0.638600
    Epoch 43, CIFAR-10 Batch 4:  Loss:     0.2710
    Accuracy: 0.643600
    Epoch 43, CIFAR-10 Batch 5:  Loss:     0.2314
    Accuracy: 0.646800
    Epoch 44, CIFAR-10 Batch 1:  Loss:     0.2763
    Accuracy: 0.645200
    Epoch 44, CIFAR-10 Batch 2:  Loss:     0.2599
    Accuracy: 0.648200
    Epoch 44, CIFAR-10 Batch 3:  Loss:     0.2357
    Accuracy: 0.637800
    Epoch 44, CIFAR-10 Batch 4:  Loss:     0.2540
    Accuracy: 0.639800
    Epoch 44, CIFAR-10 Batch 5:  Loss:     0.2198
    Accuracy: 0.645800
    Epoch 45, CIFAR-10 Batch 1:  Loss:     0.2516
    Accuracy: 0.641800
    Epoch 45, CIFAR-10 Batch 2:  Loss:     0.2284
    Accuracy: 0.643800
    Epoch 45, CIFAR-10 Batch 3:  Loss:     0.1880
    Accuracy: 0.640000
    Epoch 45, CIFAR-10 Batch 4:  Loss:     0.2301
    Accuracy: 0.651800
    Epoch 45, CIFAR-10 Batch 5:  Loss:     0.2108
    Accuracy: 0.643400
    Epoch 46, CIFAR-10 Batch 1:  Loss:     0.2676
    Accuracy: 0.641600
    Epoch 46, CIFAR-10 Batch 2:  Loss:     0.2158
    Accuracy: 0.649000
    Epoch 46, CIFAR-10 Batch 3:  Loss:     0.1938
    Accuracy: 0.639400
    Epoch 46, CIFAR-10 Batch 4:  Loss:     0.2385
    Accuracy: 0.645600
    Epoch 46, CIFAR-10 Batch 5:  Loss:     0.2036
    Accuracy: 0.651600
    Epoch 47, CIFAR-10 Batch 1:  Loss:     0.2381
    Accuracy: 0.650600
    Epoch 47, CIFAR-10 Batch 2:  Loss:     0.2394
    Accuracy: 0.647600
    Epoch 47, CIFAR-10 Batch 3:  Loss:     0.1914
    Accuracy: 0.641400
    Epoch 47, CIFAR-10 Batch 4:  Loss:     0.2320
    Accuracy: 0.644000
    Epoch 47, CIFAR-10 Batch 5:  Loss:     0.1951
    Accuracy: 0.646800
    Epoch 48, CIFAR-10 Batch 1:  Loss:     0.2197
    Accuracy: 0.643800
    Epoch 48, CIFAR-10 Batch 2:  Loss:     0.2042
    Accuracy: 0.643400
    Epoch 48, CIFAR-10 Batch 3:  Loss:     0.1731
    Accuracy: 0.652000
    Epoch 48, CIFAR-10 Batch 4:  Loss:     0.2294
    Accuracy: 0.646200
    Epoch 48, CIFAR-10 Batch 5:  Loss:     0.2097
    Accuracy: 0.653600
    Epoch 49, CIFAR-10 Batch 1:  Loss:     0.2312
    Accuracy: 0.649600
    Epoch 49, CIFAR-10 Batch 2:  Loss:     0.2296
    Accuracy: 0.645800
    Epoch 49, CIFAR-10 Batch 3:  Loss:     0.1836
    Accuracy: 0.647200
    Epoch 49, CIFAR-10 Batch 4:  Loss:     0.2477
    Accuracy: 0.635800
    Epoch 49, CIFAR-10 Batch 5:  Loss:     0.2052
    Accuracy: 0.643800
    Epoch 50, CIFAR-10 Batch 1:  Loss:     0.2350
    Accuracy: 0.648400
    Epoch 50, CIFAR-10 Batch 2:  Loss:     0.2066
    Accuracy: 0.645600
    Epoch 50, CIFAR-10 Batch 3:  Loss:     0.1757
    Accuracy: 0.644400
    Epoch 50, CIFAR-10 Batch 4:  Loss:     0.2201
    Accuracy: 0.645800
    Epoch 50, CIFAR-10 Batch 5:  Loss:     0.1965
    Accuracy: 0.647800
    Epoch 51, CIFAR-10 Batch 1:  Loss:     0.2111
    Accuracy: 0.644600
    Epoch 51, CIFAR-10 Batch 2:  Loss:     0.2176
    Accuracy: 0.640800
    Epoch 51, CIFAR-10 Batch 3:  Loss:     0.1670
    Accuracy: 0.644800
    Epoch 51, CIFAR-10 Batch 4:  Loss:     0.1956
    Accuracy: 0.648800
    Epoch 51, CIFAR-10 Batch 5:  Loss:     0.1914
    Accuracy: 0.642400
    Epoch 52, CIFAR-10 Batch 1:  Loss:     0.1971
    Accuracy: 0.644400
    Epoch 52, CIFAR-10 Batch 2:  Loss:     0.2074
    Accuracy: 0.641800
    Epoch 52, CIFAR-10 Batch 3:  Loss:     0.1613
    Accuracy: 0.645600
    Epoch 52, CIFAR-10 Batch 4:  Loss:     0.1965
    Accuracy: 0.649600
    Epoch 52, CIFAR-10 Batch 5:  Loss:     0.1840
    Accuracy: 0.647200
    Epoch 53, CIFAR-10 Batch 1:  Loss:     0.2084
    Accuracy: 0.647600
    Epoch 53, CIFAR-10 Batch 2:  Loss:     0.1914
    Accuracy: 0.640800
    Epoch 53, CIFAR-10 Batch 3:  Loss:     0.1568
    Accuracy: 0.644800
    Epoch 53, CIFAR-10 Batch 4:  Loss:     0.1944
    Accuracy: 0.649200
    Epoch 53, CIFAR-10 Batch 5:  Loss:     0.1941
    Accuracy: 0.647200
    Epoch 54, CIFAR-10 Batch 1:  Loss:     0.2289
    Accuracy: 0.639600
    Epoch 54, CIFAR-10 Batch 2:  Loss:     0.1845
    Accuracy: 0.640600
    Epoch 54, CIFAR-10 Batch 3:  Loss:     0.1704
    Accuracy: 0.644000
    Epoch 54, CIFAR-10 Batch 4:  Loss:     0.2345
    Accuracy: 0.640000
    Epoch 54, CIFAR-10 Batch 5:  Loss:     0.1836
    Accuracy: 0.647600
    Epoch 55, CIFAR-10 Batch 1:  Loss:     0.2113
    Accuracy: 0.644000
    Epoch 55, CIFAR-10 Batch 2:  Loss:     0.1782
    Accuracy: 0.644000
    Epoch 55, CIFAR-10 Batch 3:  Loss:     0.1485
    Accuracy: 0.641200
    Epoch 55, CIFAR-10 Batch 4:  Loss:     0.2055
    Accuracy: 0.639000
    Epoch 55, CIFAR-10 Batch 5:  Loss:     0.1538
    Accuracy: 0.651200
    Epoch 56, CIFAR-10 Batch 1:  Loss:     0.1811
    Accuracy: 0.653000
    Epoch 56, CIFAR-10 Batch 2:  Loss:     0.1663
    Accuracy: 0.643800
    Epoch 56, CIFAR-10 Batch 3:  Loss:     0.1414
    Accuracy: 0.646000
    Epoch 56, CIFAR-10 Batch 4:  Loss:     0.1673
    Accuracy: 0.642800
    Epoch 56, CIFAR-10 Batch 5:  Loss:     0.1513
    Accuracy: 0.648200
    Epoch 57, CIFAR-10 Batch 1:  Loss:     0.1877
    Accuracy: 0.642400
    Epoch 57, CIFAR-10 Batch 2:  Loss:     0.1632
    Accuracy: 0.641800
    Epoch 57, CIFAR-10 Batch 3:  Loss:     0.1487
    Accuracy: 0.643800
    Epoch 57, CIFAR-10 Batch 4:  Loss:     0.1802
    Accuracy: 0.636400
    Epoch 57, CIFAR-10 Batch 5:  Loss:     0.1552
    Accuracy: 0.644600
    Epoch 58, CIFAR-10 Batch 1:  Loss:     0.1829
    Accuracy: 0.647000
    Epoch 58, CIFAR-10 Batch 2:  Loss:     0.1553
    Accuracy: 0.639600
    Epoch 58, CIFAR-10 Batch 3:  Loss:     0.1353
    Accuracy: 0.645000
    Epoch 58, CIFAR-10 Batch 4:  Loss:     0.1612
    Accuracy: 0.649200
    Epoch 58, CIFAR-10 Batch 5:  Loss:     0.1379
    Accuracy: 0.647600
    Epoch 59, CIFAR-10 Batch 1:  Loss:     0.1779
    Accuracy: 0.647600
    Epoch 59, CIFAR-10 Batch 2:  Loss:     0.1660
    Accuracy: 0.644200
    Epoch 59, CIFAR-10 Batch 3:  Loss:     0.1243
    Accuracy: 0.650400
    Epoch 59, CIFAR-10 Batch 4:  Loss:     0.1540
    Accuracy: 0.649000
    Epoch 59, CIFAR-10 Batch 5:  Loss:     0.1380
    Accuracy: 0.643600
    Epoch 60, CIFAR-10 Batch 1:  Loss:     0.1857
    Accuracy: 0.644800
    Epoch 60, CIFAR-10 Batch 2:  Loss:     0.1472
    Accuracy: 0.649800
    Epoch 60, CIFAR-10 Batch 3:  Loss:     0.1314
    Accuracy: 0.645800
    Epoch 60, CIFAR-10 Batch 4:  Loss:     0.1686
    Accuracy: 0.645000
    Epoch 60, CIFAR-10 Batch 5:  Loss:     0.1324
    Accuracy: 0.649800
    Epoch 61, CIFAR-10 Batch 1:  Loss:     0.1827
    Accuracy: 0.641000
    Epoch 61, CIFAR-10 Batch 2:  Loss:     0.1623
    Accuracy: 0.644600
    Epoch 61, CIFAR-10 Batch 3:  Loss:     0.1298
    Accuracy: 0.651200
    Epoch 61, CIFAR-10 Batch 4:  Loss:     0.1636
    Accuracy: 0.643400
    Epoch 61, CIFAR-10 Batch 5:  Loss:     0.1511
    Accuracy: 0.642000
    Epoch 62, CIFAR-10 Batch 1:  Loss:     0.1581
    Accuracy: 0.638800
    Epoch 62, CIFAR-10 Batch 2:  Loss:     0.1464
    Accuracy: 0.648800
    Epoch 62, CIFAR-10 Batch 3:  Loss:     0.1168
    Accuracy: 0.646600
    Epoch 62, CIFAR-10 Batch 4:  Loss:     0.1606
    Accuracy: 0.641600
    Epoch 62, CIFAR-10 Batch 5:  Loss:     0.1295
    Accuracy: 0.648200
    Epoch 63, CIFAR-10 Batch 1:  Loss:     0.1594
    Accuracy: 0.642200
    Epoch 63, CIFAR-10 Batch 2:  Loss:     0.1563
    Accuracy: 0.650000
    Epoch 63, CIFAR-10 Batch 3:  Loss:     0.1115
    Accuracy: 0.639800
    Epoch 63, CIFAR-10 Batch 4:  Loss:     0.1576
    Accuracy: 0.647800
    Epoch 63, CIFAR-10 Batch 5:  Loss:     0.1277
    Accuracy: 0.646600
    Epoch 64, CIFAR-10 Batch 1:  Loss:     0.1630
    Accuracy: 0.639400
    Epoch 64, CIFAR-10 Batch 2:  Loss:     0.1540
    Accuracy: 0.644400
    Epoch 64, CIFAR-10 Batch 3:  Loss:     0.1135
    Accuracy: 0.640400
    Epoch 64, CIFAR-10 Batch 4:  Loss:     0.1637
    Accuracy: 0.646400
    Epoch 64, CIFAR-10 Batch 5:  Loss:     0.1279
    Accuracy: 0.648600
    Epoch 65, CIFAR-10 Batch 1:  Loss:     0.1556
    Accuracy: 0.643200
    Epoch 65, CIFAR-10 Batch 2:  Loss:     0.1384
    Accuracy: 0.644400
    Epoch 65, CIFAR-10 Batch 3:  Loss:     0.1150
    Accuracy: 0.641600
    Epoch 65, CIFAR-10 Batch 4:  Loss:     0.1539
    Accuracy: 0.638000
    Epoch 65, CIFAR-10 Batch 5:  Loss:     0.1281
    Accuracy: 0.650000
    Epoch 66, CIFAR-10 Batch 1:  Loss:     0.1637
    Accuracy: 0.640200
    Epoch 66, CIFAR-10 Batch 2:  Loss:     0.1428
    Accuracy: 0.653800
    Epoch 66, CIFAR-10 Batch 3:  Loss:     0.1071
    Accuracy: 0.644400
    Epoch 66, CIFAR-10 Batch 4:  Loss:     0.1335
    Accuracy: 0.647400
    Epoch 66, CIFAR-10 Batch 5:  Loss:     0.1269
    Accuracy: 0.649800
    Epoch 67, CIFAR-10 Batch 1:  Loss:     0.1440
    Accuracy: 0.638400
    Epoch 67, CIFAR-10 Batch 2:  Loss:     0.1370
    Accuracy: 0.640800
    Epoch 67, CIFAR-10 Batch 3:  Loss:     0.1082
    Accuracy: 0.646000
    Epoch 67, CIFAR-10 Batch 4:  Loss:     0.1344
    Accuracy: 0.647400
    Epoch 67, CIFAR-10 Batch 5:  Loss:     0.1252
    Accuracy: 0.640400
    Epoch 68, CIFAR-10 Batch 1:  Loss:     0.1485
    Accuracy: 0.640000
    Epoch 68, CIFAR-10 Batch 2:  Loss:     0.1331
    Accuracy: 0.647000
    Epoch 68, CIFAR-10 Batch 3:  Loss:     0.1168
    Accuracy: 0.638000
    Epoch 68, CIFAR-10 Batch 4:  Loss:     0.1487
    Accuracy: 0.640600
    Epoch 68, CIFAR-10 Batch 5:  Loss:     0.1344
    Accuracy: 0.646400
    Epoch 69, CIFAR-10 Batch 1:  Loss:     0.1364
    Accuracy: 0.646400
    Epoch 69, CIFAR-10 Batch 2:  Loss:     0.1286
    Accuracy: 0.642600
    Epoch 69, CIFAR-10 Batch 3:  Loss:     0.1026
    Accuracy: 0.646400
    Epoch 69, CIFAR-10 Batch 4:  Loss:     0.1465
    Accuracy: 0.638000
    Epoch 69, CIFAR-10 Batch 5:  Loss:     0.1108
    Accuracy: 0.647200
    Epoch 70, CIFAR-10 Batch 1:  Loss:     0.1384
    Accuracy: 0.645400
    Epoch 70, CIFAR-10 Batch 2:  Loss:     0.1215
    Accuracy: 0.641800
    Epoch 70, CIFAR-10 Batch 3:  Loss:     0.1033
    Accuracy: 0.649400
    Epoch 70, CIFAR-10 Batch 4:  Loss:     0.1407
    Accuracy: 0.638600
    Epoch 70, CIFAR-10 Batch 5:  Loss:     0.1062
    Accuracy: 0.647000
    Epoch 71, CIFAR-10 Batch 1:  Loss:     0.1554
    Accuracy: 0.641600
    Epoch 71, CIFAR-10 Batch 2:  Loss:     0.1250
    Accuracy: 0.641200
    Epoch 71, CIFAR-10 Batch 3:  Loss:     0.0986
    Accuracy: 0.643000
    Epoch 71, CIFAR-10 Batch 4:  Loss:     0.1333
    Accuracy: 0.641200
    Epoch 71, CIFAR-10 Batch 5:  Loss:     0.1015
    Accuracy: 0.643800
    Epoch 72, CIFAR-10 Batch 1:  Loss:     0.1538
    Accuracy: 0.632400
    Epoch 72, CIFAR-10 Batch 2:  Loss:     0.1152
    Accuracy: 0.649400
    Epoch 72, CIFAR-10 Batch 3:  Loss:     0.1011
    Accuracy: 0.646200
    Epoch 72, CIFAR-10 Batch 4:  Loss:     0.1397
    Accuracy: 0.638200
    Epoch 72, CIFAR-10 Batch 5:  Loss:     0.1120
    Accuracy: 0.642400
    Epoch 73, CIFAR-10 Batch 1:  Loss:     0.1166
    Accuracy: 0.644000
    Epoch 73, CIFAR-10 Batch 2:  Loss:     0.1250
    Accuracy: 0.637800
    Epoch 73, CIFAR-10 Batch 3:  Loss:     0.0888
    Accuracy: 0.646200
    Epoch 73, CIFAR-10 Batch 4:  Loss:     0.1475
    Accuracy: 0.639600
    Epoch 73, CIFAR-10 Batch 5:  Loss:     0.1135
    Accuracy: 0.645400
    Epoch 74, CIFAR-10 Batch 1:  Loss:     0.1193
    Accuracy: 0.653000
    Epoch 74, CIFAR-10 Batch 2:  Loss:     0.1113
    Accuracy: 0.650200
    Epoch 74, CIFAR-10 Batch 3:  Loss:     0.0941
    Accuracy: 0.646200
    Epoch 74, CIFAR-10 Batch 4:  Loss:     0.1470
    Accuracy: 0.633400
    Epoch 74, CIFAR-10 Batch 5:  Loss:     0.1081
    Accuracy: 0.645000
    Epoch 75, CIFAR-10 Batch 1:  Loss:     0.1306
    Accuracy: 0.644400
    Epoch 75, CIFAR-10 Batch 2:  Loss:     0.1100
    Accuracy: 0.643200
    Epoch 75, CIFAR-10 Batch 3:  Loss:     0.0893
    Accuracy: 0.646200
    Epoch 75, CIFAR-10 Batch 4:  Loss:     0.1299
    Accuracy: 0.637000
    Epoch 75, CIFAR-10 Batch 5:  Loss:     0.1108
    Accuracy: 0.645000
    Epoch 76, CIFAR-10 Batch 1:  Loss:     0.1293
    Accuracy: 0.635800
    Epoch 76, CIFAR-10 Batch 2:  Loss:     0.1143
    Accuracy: 0.645400
    Epoch 76, CIFAR-10 Batch 3:  Loss:     0.0916
    Accuracy: 0.638600
    Epoch 76, CIFAR-10 Batch 4:  Loss:     0.1400
    Accuracy: 0.635400
    Epoch 76, CIFAR-10 Batch 5:  Loss:     0.0966
    Accuracy: 0.647600
    Epoch 77, CIFAR-10 Batch 1:  Loss:     0.1311
    Accuracy: 0.646000
    Epoch 77, CIFAR-10 Batch 2:  Loss:     0.0966
    Accuracy: 0.648400
    Epoch 77, CIFAR-10 Batch 3:  Loss:     0.0824
    Accuracy: 0.642600
    Epoch 77, CIFAR-10 Batch 4:  Loss:     0.1455
    Accuracy: 0.636600
    Epoch 77, CIFAR-10 Batch 5:  Loss:     0.1105
    Accuracy: 0.646200
    Epoch 78, CIFAR-10 Batch 1:  Loss:     0.1304
    Accuracy: 0.639200
    Epoch 78, CIFAR-10 Batch 2:  Loss:     0.1124
    Accuracy: 0.641600
    Epoch 78, CIFAR-10 Batch 3:  Loss:     0.0948
    Accuracy: 0.644400
    Epoch 78, CIFAR-10 Batch 4:  Loss:     0.1098
    Accuracy: 0.647200
    Epoch 78, CIFAR-10 Batch 5:  Loss:     0.0951
    Accuracy: 0.646000
    Epoch 79, CIFAR-10 Batch 1:  Loss:     0.1276
    Accuracy: 0.642200
    Epoch 79, CIFAR-10 Batch 2:  Loss:     0.1163
    Accuracy: 0.639400
    Epoch 79, CIFAR-10 Batch 3:  Loss:     0.1029
    Accuracy: 0.641400
    Epoch 79, CIFAR-10 Batch 4:  Loss:     0.1206
    Accuracy: 0.644200
    Epoch 79, CIFAR-10 Batch 5:  Loss:     0.1067
    Accuracy: 0.646000
    Epoch 80, CIFAR-10 Batch 1:  Loss:     0.1045
    Accuracy: 0.650000
    Epoch 80, CIFAR-10 Batch 2:  Loss:     0.1078
    Accuracy: 0.645600
    Epoch 80, CIFAR-10 Batch 3:  Loss:     0.0914
    Accuracy: 0.639600
    Epoch 80, CIFAR-10 Batch 4:  Loss:     0.1260
    Accuracy: 0.638600
    Epoch 80, CIFAR-10 Batch 5:  Loss:     0.0965
    Accuracy: 0.643600
    Epoch 81, CIFAR-10 Batch 1:  Loss:     0.0976
    Accuracy: 0.649800
    Epoch 81, CIFAR-10 Batch 2:  Loss:     0.1145
    Accuracy: 0.642200
    Epoch 81, CIFAR-10 Batch 3:  Loss:     0.0889
    Accuracy: 0.642400
    Epoch 81, CIFAR-10 Batch 4:  Loss:     0.0994
    Accuracy: 0.641400
    Epoch 81, CIFAR-10 Batch 5:  Loss:     0.0971
    Accuracy: 0.647200
    Epoch 82, CIFAR-10 Batch 1:  Loss:     0.1189
    Accuracy: 0.645200
    Epoch 82, CIFAR-10 Batch 2:  Loss:     0.1055
    Accuracy: 0.639800
    Epoch 82, CIFAR-10 Batch 3:  Loss:     0.1099
    Accuracy: 0.643000
    Epoch 82, CIFAR-10 Batch 4:  Loss:     0.1160
    Accuracy: 0.640600
    Epoch 82, CIFAR-10 Batch 5:  Loss:     0.0981
    Accuracy: 0.642200
    Epoch 83, CIFAR-10 Batch 1:  Loss:     0.1286
    Accuracy: 0.638400
    Epoch 83, CIFAR-10 Batch 2:  Loss:     0.1067
    Accuracy: 0.640200
    Epoch 83, CIFAR-10 Batch 3:  Loss:     0.0995
    Accuracy: 0.642200
    Epoch 83, CIFAR-10 Batch 4:  Loss:     0.1063
    Accuracy: 0.643000
    Epoch 83, CIFAR-10 Batch 5:  Loss:     0.0921
    Accuracy: 0.651200
    Epoch 84, CIFAR-10 Batch 1:  Loss:     0.1054
    Accuracy: 0.649800
    Epoch 84, CIFAR-10 Batch 2:  Loss:     0.0995
    Accuracy: 0.642400
    Epoch 84, CIFAR-10 Batch 3:  Loss:     0.0853
    Accuracy: 0.641400
    Epoch 84, CIFAR-10 Batch 4:  Loss:     0.1148
    Accuracy: 0.644000
    Epoch 84, CIFAR-10 Batch 5:  Loss:     0.0890
    Accuracy: 0.645200
    Epoch 85, CIFAR-10 Batch 1:  Loss:     0.1156
    Accuracy: 0.646600
    Epoch 85, CIFAR-10 Batch 2:  Loss:     0.1030
    Accuracy: 0.647200
    Epoch 85, CIFAR-10 Batch 3:  Loss:     0.0781
    Accuracy: 0.635200
    Epoch 85, CIFAR-10 Batch 4:  Loss:     0.1015
    Accuracy: 0.641600
    Epoch 85, CIFAR-10 Batch 5:  Loss:     0.0850
    Accuracy: 0.645400
    Epoch 86, CIFAR-10 Batch 1:  Loss:     0.0958
    Accuracy: 0.648600
    Epoch 86, CIFAR-10 Batch 2:  Loss:     0.0915
    Accuracy: 0.644400
    Epoch 86, CIFAR-10 Batch 3:  Loss:     0.0823
    Accuracy: 0.637600
    Epoch 86, CIFAR-10 Batch 4:  Loss:     0.1023
    Accuracy: 0.643800
    Epoch 86, CIFAR-10 Batch 5:  Loss:     0.0944
    Accuracy: 0.646800
    Epoch 87, CIFAR-10 Batch 1:  Loss:     0.1135
    Accuracy: 0.642400
    Epoch 87, CIFAR-10 Batch 2:  Loss:     0.0949
    Accuracy: 0.644800
    Epoch 87, CIFAR-10 Batch 3:  Loss:     0.0848
    Accuracy: 0.641400
    Epoch 87, CIFAR-10 Batch 4:  Loss:     0.0948
    Accuracy: 0.641000
    Epoch 87, CIFAR-10 Batch 5:  Loss:     0.0885
    Accuracy: 0.640800
    Epoch 88, CIFAR-10 Batch 1:  Loss:     0.0974
    Accuracy: 0.653400
    Epoch 88, CIFAR-10 Batch 2:  Loss:     0.0850
    Accuracy: 0.645400
    Epoch 88, CIFAR-10 Batch 3:  Loss:     0.0773
    Accuracy: 0.646800
    Epoch 88, CIFAR-10 Batch 4:  Loss:     0.0935
    Accuracy: 0.642800
    Epoch 88, CIFAR-10 Batch 5:  Loss:     0.0892
    Accuracy: 0.643600
    Epoch 89, CIFAR-10 Batch 1:  Loss:     0.0952
    Accuracy: 0.646600
    Epoch 89, CIFAR-10 Batch 2:  Loss:     0.0907
    Accuracy: 0.640400
    Epoch 89, CIFAR-10 Batch 3:  Loss:     0.0748
    Accuracy: 0.644600
    Epoch 89, CIFAR-10 Batch 4:  Loss:     0.0819
    Accuracy: 0.639000
    Epoch 89, CIFAR-10 Batch 5:  Loss:     0.0956
    Accuracy: 0.644800
    Epoch 90, CIFAR-10 Batch 1:  Loss:     0.0865
    Accuracy: 0.648400
    Epoch 90, CIFAR-10 Batch 2:  Loss:     0.0927
    Accuracy: 0.638000
    Epoch 90, CIFAR-10 Batch 3:  Loss:     0.0713
    Accuracy: 0.638400
    Epoch 90, CIFAR-10 Batch 4:  Loss:     0.0867
    Accuracy: 0.644200
    Epoch 90, CIFAR-10 Batch 5:  Loss:     0.0855
    Accuracy: 0.646600
    Epoch 91, CIFAR-10 Batch 1:  Loss:     0.0994
    Accuracy: 0.637400
    Epoch 91, CIFAR-10 Batch 2:  Loss:     0.0858
    Accuracy: 0.640200
    Epoch 91, CIFAR-10 Batch 3:  Loss:     0.0784
    Accuracy: 0.643000
    Epoch 91, CIFAR-10 Batch 4:  Loss:     0.0849
    Accuracy: 0.638000
    Epoch 91, CIFAR-10 Batch 5:  Loss:     0.0871
    Accuracy: 0.647600
    Epoch 92, CIFAR-10 Batch 1:  Loss:     0.0916
    Accuracy: 0.644600
    Epoch 92, CIFAR-10 Batch 2:  Loss:     0.0733
    Accuracy: 0.645600
    Epoch 92, CIFAR-10 Batch 3:  Loss:     0.0712
    Accuracy: 0.639200
    Epoch 92, CIFAR-10 Batch 4:  Loss:     0.0814
    Accuracy: 0.646000
    Epoch 92, CIFAR-10 Batch 5:  Loss:     0.0808
    Accuracy: 0.645400
    Epoch 93, CIFAR-10 Batch 1:  Loss:     0.0920
    Accuracy: 0.643000
    Epoch 93, CIFAR-10 Batch 2:  Loss:     0.0800
    Accuracy: 0.645400
    Epoch 93, CIFAR-10 Batch 3:  Loss:     0.0669
    Accuracy: 0.639800
    Epoch 93, CIFAR-10 Batch 4:  Loss:     0.0856
    Accuracy: 0.634200
    Epoch 93, CIFAR-10 Batch 5:  Loss:     0.0888
    Accuracy: 0.641200
    Epoch 94, CIFAR-10 Batch 1:  Loss:     0.0891
    Accuracy: 0.643000
    Epoch 94, CIFAR-10 Batch 2:  Loss:     0.0797
    Accuracy: 0.644400
    Epoch 94, CIFAR-10 Batch 3:  Loss:     0.0613
    Accuracy: 0.645200
    Epoch 94, CIFAR-10 Batch 4:  Loss:     0.0875
    Accuracy: 0.637800
    Epoch 94, CIFAR-10 Batch 5:  Loss:     0.0848
    Accuracy: 0.642600
    Epoch 95, CIFAR-10 Batch 1:  Loss:     0.0824
    Accuracy: 0.644000
    Epoch 95, CIFAR-10 Batch 2:  Loss:     0.0818
    Accuracy: 0.636400
    Epoch 95, CIFAR-10 Batch 3:  Loss:     0.0689
    Accuracy: 0.637800
    Epoch 95, CIFAR-10 Batch 4:  Loss:     0.0829
    Accuracy: 0.637000
    Epoch 95, CIFAR-10 Batch 5:  Loss:     0.0779
    Accuracy: 0.645200
    Epoch 96, CIFAR-10 Batch 1:  Loss:     0.0971
    Accuracy: 0.642400
    Epoch 96, CIFAR-10 Batch 2:  Loss:     0.0904
    Accuracy: 0.640200
    Epoch 96, CIFAR-10 Batch 3:  Loss:     0.0733
    Accuracy: 0.637600
    Epoch 96, CIFAR-10 Batch 4:  Loss:     0.0846
    Accuracy: 0.637200
    Epoch 96, CIFAR-10 Batch 5:  Loss:     0.0699
    Accuracy: 0.639600
    Epoch 97, CIFAR-10 Batch 1:  Loss:     0.0889
    Accuracy: 0.644800
    Epoch 97, CIFAR-10 Batch 2:  Loss:     0.0786
    Accuracy: 0.635800
    Epoch 97, CIFAR-10 Batch 3:  Loss:     0.0856
    Accuracy: 0.641400
    Epoch 97, CIFAR-10 Batch 4:  Loss:     0.0821
    Accuracy: 0.633000
    Epoch 97, CIFAR-10 Batch 5:  Loss:     0.0623
    Accuracy: 0.645000
    Epoch 98, CIFAR-10 Batch 1:  Loss:     0.0856
    Accuracy: 0.646600
    Epoch 98, CIFAR-10 Batch 2:  Loss:     0.0764
    Accuracy: 0.646800
    Epoch 98, CIFAR-10 Batch 3:  Loss:     0.0556
    Accuracy: 0.640200
    Epoch 98, CIFAR-10 Batch 4:  Loss:     0.0802
    Accuracy: 0.630600
    Epoch 98, CIFAR-10 Batch 5:  Loss:     0.0740
    Accuracy: 0.642800
    Epoch 99, CIFAR-10 Batch 1:  Loss:     0.0793
    Accuracy: 0.644600
    Epoch 99, CIFAR-10 Batch 2:  Loss:     0.0794
    Accuracy: 0.644600
    Epoch 99, CIFAR-10 Batch 3:  Loss:     0.0755
    Accuracy: 0.645000
    Epoch 99, CIFAR-10 Batch 4:  Loss:     0.0795
    Accuracy: 0.634400
    Epoch 99, CIFAR-10 Batch 5:  Loss:     0.0820
    Accuracy: 0.644200
    Epoch 100, CIFAR-10 Batch 1:  Loss:     0.0880
    Accuracy: 0.643800
    Epoch 100, CIFAR-10 Batch 2:  Loss:     0.0739
    Accuracy: 0.640800
    Epoch 100, CIFAR-10 Batch 3:  Loss:     0.0666
    Accuracy: 0.640400
    Epoch 100, CIFAR-10 Batch 4:  Loss:     0.0792
    Accuracy: 0.633800
    Epoch 100, CIFAR-10 Batch 5:  Loss:     0.0804
    Accuracy: 0.645000


# Checkpoint
The model has been saved to disk.
## Test Model
Test your model against the test dataset.  This will be your final accuracy. You should have an accuracy greater than 50%. If you don't, keep tweaking the model architecture and parameters.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import tensorflow as tf
import pickle
import helper
import random

# Set batch size if not already set
try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = './image_classification'
n_samples = 4
top_n_predictions = 3

def test_model():
    """
    Test the saved model against the test dataset
    """

    test_features, test_labels = pickle.load(open('preprocess_test.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for test_feature_batch, test_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: test_feature_batch, loaded_y: test_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


test_model()
```

    INFO:tensorflow:Restoring parameters from ./image_classification
    Testing Accuracy: 0.6417566657066345
    



![png](output_36_1.png)


## Why 50-80% Accuracy?
You might be wondering why you can't get an accuracy any higher. First things first, 50% isn't bad for a simple CNN.  Pure guessing would get you 10% accuracy. However, you might notice people are getting scores [well above 80%](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130).  That's because we haven't taught you all there is to know about neural networks. We still need to cover a few more techniques.
## Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook.  Save the notebook file as "dlnd_image_classification.ipynb" and save it as a HTML file under "File" -> "Download as".  Include the "helper.py" and "problem_unittests.py" files in your submission.
