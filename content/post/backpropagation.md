+++
title = "Breaking down Neural Networks: An intuitive approach to backprop!"
date = 2018-06-16T23:22:47+05:30
draft = false
summary = "Whether you’re a beginner in machine learning or an expert, you would have had a hard time understanding the concept of backpropagation in neural networks.This article aims to explain the intuition behind the backpropagation algorithm in a simple way, that can be understandable even by a machine learning beginner."
# Authors. Comma separated list, e.g. `["Bob Smith", "David Jones"]`.
authors = ["Benedict Florance"]

# Tags and categories
# For example, use `tags = []` for no tags, or the form `tags = ["A Tag", "Another Tag"]` for one or more tags.
tags = ['machine_learning', 'deep_learning', 'neural_networks', 'backpropagation', 'AI']
categories = []

# Featured image
# Place your image in the `static/img/` folder and reference its filename below, e.g. `image = "example.jpg"`.
# Use `caption` to display an image caption.
#   Markdown linking is allowed, e.g. `caption = "[Image credit](http://example.org)"`.
# Set `preview` to `false` to disable the thumbnail in listings.
[header]
image = "backpropagation.jpg"
caption = ""
preview = true
+++

Whether you’re a beginner in machine learning or an expert, you would have had a hard time understanding the concept of backpropagation in neural networks. If you’re a beginner, the first look of the complex steps strung together in the backprop algorithm would’ve been definitely daunting. While a few of us would have spent time in analysing the algorithm and getting the intuition, most of us would have abstracted the learning process. A common and a very sensible question that can arise for people who have worked with deep learning frameworks like Facebook’s Pytorch and Google’s Tensorflow is:

> "Why do we have to spend time in understanding the necessity of backprop if a single line of `loss.backward()` in Pytorch or `tape.gradient(loss_value, model.variables)` in TensorFlow can do the magic?"

A one-liner. You can’t debug your neural network effectively and won’t be able to figure out where you are going wrong!

This article aims to explain the intuition behind the backpropagation algorithm in a simple way, that can be understandable even by a machine learning beginner.

### A short introduction to neural networks:

![A neural network with two hidden layers. Source: [www.towardsdatascience.com](http://www.towardsdatascience.com)](https://cdn-images-1.medium.com/max/2000/1*WNxN2ArLaGt0-Rm3tzWw1g.jpeg)*A neural network with two hidden layers. Source: [www.towardsdatascience.com](http://www.towardsdatascience.com)*

> The core idea of neural networks is to compute weighted sums of the values in the input layer and create a mapping between the input and output layer by a series of functions (in general, nonlinear functions).

Sounds complex? Let me break it down. Every neural network has an input layer, a series of hidden layers and an output layer. Let us take the task of classifying our images into four categories ‘cat’, ‘dog’, ‘frog’ and ‘horse’. The values in the **input layer** represent the pixel values of a given image that we want to classify into four categories. The small circles in each layer are called **neurons**. The values of the **output layer** represent the score for each category. We classify the image into the category that gets the highest score. For instance, if the ‘frog’ neuron in the output layer receives the highest value in comparison to the other neurons of the layer, we say the image is a ‘frog’. The other intermediary layers are called **hidden layers.**

![A closer look at a single neuron. Source: Google Images](https://cdn-images-1.medium.com/max/2000/0*4Je3OlrLh-1QIv_V.png)*A closer look at a single neuron. Source: Google Images*

Every junction between two layers have a set of parameters called **weights**. These weights are not random numbers, the weight matrix between different layers can be visualized as a template or a feature that we are looking for in the image to classify it. The values of the next layer are calculated by applying a function called **activation function** to the values of the previous layer and the weights in between the two layers. Commonly used activation functions are sigmoid, tanh, Rectified Linear Unit (also called ReLU), leaky ReLU and Maxout. The difference between the output that we predict using the network and the actual class of the image determines the loss of the network. Greater the number of images that we classify correctly, lesser the loss. There are several ways of computing the loss of a neural network. One naive approach would be to find the mean squared error i.e, the mean of squares of the difference between the predicted and actual values. Other techniques that are often used to compute loss are softmax(cross-entropy) and support vector machine (hinge) loss. So, the aim of a neural network problem is to learn the best set of weights to give us the desired scores in the output layer. In other words, to minimize the loss function of the network.

## What is backpropagation?

Training a neural network typically happens in two phases.

1. **Forward Pass:** We compute the outputs of every node in the forward pass and calculate the final loss of the network.

1. **Backward Pass:** We start at the end of the network, backpropagate or feed the errors back, recursively apply chain rule to compute gradients all the way to the inputs of the network and then update the weights. This method of backpropagating the errors and computing the gradients is called **backpropagation.**

It is a very popular neural network training algorithm as it is conceptually clear, computationally tractable and produces optimal results in general. To reiterate, the aim of a typical neural network problem is to discover a model that fits our data ‘best’. Ultimately, we want to minimize the cost or loss function by choosing the best set of parameters.

### A short recap of derivatives:

Let us consider the following function.

![Product of two variables.](https://cdn-images-1.medium.com/max/2000/1*haSLzvmimPyGftWBixHxUA.png)*Product of two variables.*

![Partial derivatives of the function w.r.t the inputs.](https://cdn-images-1.medium.com/max/2000/1*srZ0Vnma4URroyKbeYAGnA.png)*Partial derivatives of the function w.r.t the inputs.*

It is simple enough to find the partial derivative with respect to either of the inputs. For example, if

![Let us take sample values as inputs to the function.](https://cdn-images-1.medium.com/max/2000/1*f2g7p9B3QAaIo8rO8v265w.png)*Let us take sample values as inputs to the function.*

The partial derivative on variable x ( ∂f/ ∂x) and variable y ( ∂f/ ∂y) are 3 and -2 respectively. This gives us an understanding that increasing the variable x by an amount ε would increase the output function by 3ε. Similarly, increasing the variable y by an amount ε would decrease the output function by 2ε. Thus, the derivative of a function on each variable tells us the sensitivity of the function with respect to that variable.

Drawing a parallel analogy, by computing the gradients of the loss function with respect to the weights and the inputs of the neural network, we can determine the sensitivity of the loss function with respect to these parameters. These gradients are a measure of how well the neural network is performing and how the parameters of the model are affecting the loss function. It also helps us in fine tuning the weights of the network to minimize our loss and find a model that fits our data.

## How to backpropagate?

We’ll be using the following example throughout the rest of the article.

![This example will be used throughout the rest of the article.](https://cdn-images-1.medium.com/max/2000/1*DQ5xxpPVU44Lqddo8f7Yhw.png)*This example will be used throughout the rest of the article.*

It’s a good practice to draw computation graphs and analyse the expressions, albeit easier only for simple expressions. So, let’s draw one. We also introduce intermediary variables like x and y to make our calculations simpler.

![Computational graph for the example f=(a+b)(b+c) with a = -1, b = 3 and c = 4.](https://cdn-images-1.medium.com/max/2000/1*azqHvbrNsZ8AIZ7H75tbIQ.jpeg)*Computational graph for the example f=(a+b)(b+c) with a = -1, b = 3 and c = 4.*

**Every node in a computational graph can compute two things — the output of the node and the local gradient of the node without even being aware of the rest of the graph.** Local gradients of a node are the derivatives of the output of the node with respect to each of the inputs.

![Local gradients of the nodes in the computational graph.](https://cdn-images-1.medium.com/max/2000/1*7XxBjQzyLCkWKEgJD_w9jQ.png)*Local gradients of the nodes in the computational graph.*

We have marked the outputs on the graph and have also calculated the local gradients of the nodes. **Backpropagation is a “local” process and can be viewed as a recursive application of the chain rule.** Now, we want the sensitivity of our output (loss) function w.r.t to the input variables a, b and c of the graph (i.e. ∂f/ ∂a, ∂f/ ∂b and ∂f/ ∂c). We start with the output variable and find the derivative of the output of the graph w.r.t to every variable by recursive chain rule. The derivative of the output variable w.r.t itself is one.

![Backpropagating and finding the gradient of f w.r.t all the variables in the graph by the application of chain rule. Blue elements represent the outputs of the nodes whereas red element represents the gradients that were calculated during backpropagation. (Note that f = (a+b)(b+c), x= a+b and y = b+c)](https://cdn-images-1.medium.com/max/2338/1*GEpvvmhoj0yRTi_kpDS6Eg.png)*Backpropagating and finding the gradient of f w.r.t all the variables in the graph by the application of chain rule. Blue elements represent the outputs of the nodes whereas red element represents the gradients that were calculated during backpropagation. (Note that f = (a+b)(b+c), x= a+b and y = b+c)*

Let us take one node in the graph and get a clear picture.

![A closer look of backpropagation — considering a single node.](https://cdn-images-1.medium.com/max/2000/1*mudCF1PDeBlErRNOPXD24Q.jpeg)*A closer look of backpropagation - considering a single node.*

We computed the outputs and local gradients of every node before we started backpropagating. The graph which was unaware of the existence of other nodes when we calculated the outputs and local gradients, interacts with the other nodes and learns the derivative of its output value on the final output of the graph ( ∂f/ ∂x). Thus, **we find the derivative of the output of the graph w.r.t a variable (** ∂f/ ∂a**) by multiplying its local gradient (** ∂x/ ∂a) **with the upstream gradient that we receive from the node’s output value. (** ∂f/ ∂x)**. This is the essence of backpropagation**. If we look at variable b, we can use the multivariate chain rule to find the derivative of the output f w.r.t the variable b.

![Gradients add at branches — multivariate chain rule.](https://cdn-images-1.medium.com/max/2000/1*wT88MRBXEztY5KrRq_PzVg.png)*Gradients add at branches — multivariate chain rule.*

We can also interpret this multivariate chain rule by saying “**gradients add at branches**”. Thus, we have found the sensitivity of variables a, b and c on the output f by computing the derivatives through backpropagation.

At this point, you might have the following questions.

> “Why this roundabout method of finding gradients by backpropagation while we can compute the same gradients by differentiating in a simple, forward manner?”

> “Oh, backpropagation is nothing but chain rule! What’s so special in that?”

## Why backpropagation?

Let us look at two strategies by which we can compute gradients.

### Strategy 1: Forward differentiation

It is the usual way of finding gradients, the way that we all learnt in our high school. Let us consider the same example again. Without loss of generality, we choose variable b and find gradients upwards.

![Forward differentiation — the traditional way. (Note that f = (a+b)(b+c), x= a+b and y = b+c)](https://cdn-images-1.medium.com/max/2268/1*r5bFLL1rKCrMuZOnZR0lcg.png)*Forward differentiation — the traditional way. (Note that f = (a+b)(b+c), x= a+b and y = b+c)*

![](https://cdn-images-1.medium.com/max/2000/1*wLd-zTw71u7GDnsgjKwKoA.png)

Thus, we have computed the derivative of f (our output) w.r.t. variable b (one of the inputs).

Forward differentiation determines how one of the inputs affect every node in the graph.

## Strategy 2: Reverse Differentiation:

We already implemented reverse differentiation when we learnt how to do backpropagation. Just to have a recap, let’s look at the graph without any chain-rule steps written on the graph.

![Backpropagation or reverse differentiation. Refer the diagram under 'How to Backpropagate?' for the chain-rule steps.](https://cdn-images-1.medium.com/max/2360/1*U3mVDYuvnaLhJzIFw_d5qQ.png)*Backpropagation or reverse differentiation. Refer the diagram under 'How to Backpropagate?' for the chain-rule steps.*

If you notice properly, by doing reverse differentiation (or backpropagation), we have computed the derivative of f (our output or loss function) with respect to every node in the graph. **Yeah, you saw that right, with respect to every single node in the graph!**

> Forward-mode differentiation gave us the derivative of our output with respect to one of the inputs, but reverse-mode differentiation gives us all of them.

As we have only three variables as input to the graph, we can see a thrice speedup by performing backpropagation (reverse differentiation) instead of forward differentiation.

> Why thrice speedup?

![](https://cdn-images-1.medium.com/max/2000/1*9tfrBe7pAtqEiVwz5w7j3A.png)

> We found only the derivative of f w.r.t b in forward differentiation.

![](https://cdn-images-1.medium.com/max/2000/1*Ua0MG2skPgLovioPghKm5g.png)

> Whereas, we found the derivative of f w.r.t all the three input variables, by backprop in one fell swoop.

### So, is that all?

![Gradient Descent. J(w) represents the loss and w represents the weight. Source: Google Images](https://cdn-images-1.medium.com/max/600/0*rBQI7uBhBKE8KT-X.png)*Gradient Descent. J(w) represents the loss and w represents the weight. Source: Google Images*

To reiterate, loss function quantifies the quality of our weights. Having calculated the gradients of our loss function with respect to all the parameters of the neural networks, its time to update the model parameters using these gradients to make our model more fit to the data. A commonly used technique to optimize weight parameters is **gradient descent**. In gradient descent, we take small baby steps in the direction of the minima to get optimized weight parameters. The size of the steps that we take to reach the optimum value is determined by a parameter called **learning rate.** Other commonly used techniques for weight updation are AdaGrad, RMSProp and Adam optimization. Thus, by making use of the gradients computed through an efficient backprop, we are able to find the best set of weights that minimizes our loss function. We do this by backpropagating the neural network multiple times until we reach a steady loss.

## Yo, backprop is powerful!

Convolutional Neural Networks (CNNs) are a class of deep neural networks (deep, implying large number of hidden layers) that are primarily used for visual recognition — classifying images. ImageNet, the largest visual database consists of over 14 million images belonging to 20 thousand categories. It is a common practice to evaluate the performance of a CNN by training and testing it on the ImageNet database due to the large number of labeled images that it has. The current standard CNN architecture that performs the best on ImageNet is ResNet-152, which has 152 layers and a parameter count close to a billion! By performing backpropagation, we can get the gradients of the loss function with respect to all the inputs and weights of the network. Think of the massive speedup of billion times when we choose backpropagation over forward differentiation. Sounds awesome, doesn’t it?

Hope you understood the actual intuition behind backpropagation and why it is preferred. Cheers! :)

### References:

1. [http://cs231n.github.io/optimization-2/](http://cs231n.github.io/optimization-2/)

1. [https://www.youtube.com/watch?v=d14TUNcbn1k](https://www.youtube.com/watch?v=d14TUNcbn1k)

1. [http://colah.github.io/posts/2015-08-Backprop/](http://colah.github.io/posts/2015-08-Backprop/)

1. [https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)

### More reading:

1. [http://neuralnetworksanddeeplearning.com/chap2.html](http://neuralnetworksanddeeplearning.com/chap2.html)

1. [http://cs231n.stanford.edu/handouts/derivatives.pdf](http://cs231n.stanford.edu/handouts/derivatives.pdf)

P. S. Used LaTeX and [www.draw.io](http://www.draw.io) for the network diagrams.