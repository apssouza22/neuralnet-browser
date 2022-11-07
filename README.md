# Vanilla Neural Network on the browser with Javascript

This project shows how to train a neural network on the browser using Vanilla Neural Network(No library) 
then we compare its performance with a neural network trained using Tensorflow.js.

The model is trained on the MNIST dataset, which contains images of handwritten digits. 

This project lets you train a handwritten digit recognizer using three different model approaches:
- Fully Connected Neural Network - Vanilla Artificial Neural Network(My own implementation)
- Fully Connected Neural Network (also known as a DenseNet) Using TensorFlow.js
- Convolutional Neural Network(also known as a ConvNet or CNN) Using TensorFlow.js

Note: currently the entire dataset of MNIST images is stored in a PNG image

**If you want to learn more about Deep learning and Neural Networks, I recommend you to check out my [Free Deep Learning Course](https://www.udemy.com/course/convolutional-neural-net-cnn-for-developers/)**

![Alt text](predictions.png?raw=true "inference digits")

## Free course
I have created a free course covering all the component implementation in this project.
[Deep Learning: Neural Networks in Javascript from scratch](https://www.udemy.com/course/deep-learning-neural-networks-in-javascript-from-scratch/?referralCode=8609C3432BD37D794205).

## Getting Started
run `yarn install` to install all the dependencies.

run `yarn watch` to start the development server


### If this project helped you, consider leaving a star  and by me a coffee
<a href="https://www.buymeacoffee.com/apssouza"><img src="https://miro.medium.com/max/654/1*rQv8JgstmK0juxP-Kb4IGg.jpeg"></a>

## Implementing a Backpropagation algorithm from scratch
In this repository, you will learn how to implement the backpropagation algorithm from scratch using Javascript.

What is Backpropagation? Back-propagation is the essence of neural net training. 
It is the method of fine-tuning the weights of a neural net based on the error rate obtained in the previous epoch 
(i.e., iteration). Proper tuning of the weights allows you to reduce error rates and to make the model reliable by increasing its generalization.

Backpropagation is a short form for "backward propagation of errors." It is a standard method of training artificial neural 
networks. This method helps to calculate the gradient of a loss function with respects to all the weights in the network.

The backpropagation algorithm consists of two phases:

* The forward pass where we pass our inputs through the network to obtain our output classifications.
* The backward pass (i.e., weight update phase) where we compute the gradient of the loss function and use this information to iteratively apply 
the chain rule to update the weights in our network.

<img src="https://camo.githubusercontent.com/97ba6d4d96e26e783e836e035828b17ab314488b5983a9b94a58856c4d050c13/68747470733a2f2f7777772e6775727539392e636f6d2f696d616765732f312f3033303831395f303933375f4261636b50726f70616761312e706e67">