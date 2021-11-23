---
layout: post
title: Mobility in Deep Learning Is a Big Optimization Problem, and We've Solved It! (Almost) - Blog Post as Conference Contributions
tags: [Deep Learning, Deep Compression, ICLR, Proposal, call]
authors: Siga, Carolene, Carnegie Mellon University; Srivastava, Shreya, Carnegie Mellon University;  Siga, Davidson, Carnegie Mellon University
---

# **Mobility in Deep Learning Is a Big Optimization Problem, and We've Solved It! (Almost)**

<p align="center">
  <img src="MLLD_blog/public/images/b235fb4a4d6d55e74a67dc988814630a.jpeg">
</p>

*“Deep learning will radically change the ways we interact with technology.”*  
\-Harvard Business Review [1]

>Have you ever thought of how your phone looks at you and decides to unlock in an
instant?

or

>Have you wondered how Alexa, Google Assistant, or Siri can continue to recognize
you even offline?

### **Deep learning on Mobile makes this possible!**

By the way, this isn't restricted to just mobile phones; it can be and is being
extended to Drones, Robots, Smart Glasses, Self-Driving Cars, and more.

### **What is Deep Learning? Why is it useful here?**

DL is the next wave of Artificial Intelligence and will pave the way for
machines to learn the real world accurately. It was inspired by the structure of
the human brain and can accurately understand complex relations in the real
world.

The learning begins with the Neuron, the most fundamental element of DL
architecture. Unlike our brain, with a multi-layered massive network with
infinite neurons. A DL architecture comprises finite layers of these neurons,
which are interconnected and collectively learn complex relations on the data
they observe.

Below is an example of a Deep Neural Network (the architecture used for DL)
<center>
<img width =500 align="center" src="MLLD_blog/public/images/9682a9c5b4152a4addc326a3f433db66.png">

[Figure 1.1 ](<https://www.researchgate.net/figure/Deep-Neural-Network-architecture_fig1_330120030>)

</center>
<br/>

Source:


Historically, machines executed tasks by being coded with deterministic
algorithms, which had to be coded by a human. DL helps us by not needing any
algorithm to solve the problem in the first place. DL can develop a solution
itself by learning the algorithm's pattern itself.

### **Why do we need to pick DL instead of ML or other Algorithms?**

DL has proved to be better than traditional Machine Learning or Deterministic
Solutions towards learning from real-world data.

"*Deep Learning is gaining much popularity due to its supremacy in terms of
accuracy when trained with a huge amount of data"* [2]

<center>
<img align="center" width=400 src="MLLD_blog/public/images/f540bcba2c39a89657f268fa86588296.png">

[Figure 1.2](https://www.researchgate.net/figure/Illustrative-comparison-between-performance-of-deep-learning-against-that-of-most-other_fig2_336078673)

<br/>
<img align="center" width=400 src="MLLD_blog/MLLD_blog/public/images/f0714079aa20da7ee34eee92a5cdb557.png">

[Figure 1.3](https://trends.google.com/trends/explore?date=2005-10-21%202019-12-31&q=%2Fm%2F0h1fn8h)

<br/>
</center>
<br/>

Research Gate (Figure 1.2) shows DL beats other Algorithms significantly when
the number of data increases. We also Notice, the popularity of DL has increased
substantially (Figure 1.3) over time.

And In Twenty-First Century, Data never sleeps.

<center>

<img align="center" width=300 src="MLLD_blog/MLLD_blog/public/images/1e7e591ac75b717e79bb1d6aeed2ad8c.jpeg">

[Figure 1.4](https://www.visualcapitalist.com/wp-content/uploads/2019/07/big-data-getting-bigger.jpg)

</center>


### **The Problem with Deep Learning**

We know Deep Learning is accurate, but accuracy comes at the cost of
implementation.

The best DL models require the machines to have high computational power and
memory. The smaller mobile devices like smartphones, drones, etc., do not have
this ability and hence face a challenge in access to Deep Learning Models.

Running DL on Mobile can cause:

1. Mobile Apps need to be small, but DL causes **developers to suffer** because
    storing DL becomes **heavy in memory**.

2. Mobile hardware suffers due to **heavy computations** used by DL models on
    mobile, causing the **Hardware engineer to suffer** while planning the app's
    interaction with hardware.

Many would say, why not deploy the model on the cloud instead of local hardware.

But, running it on the cloud makes the whole process inefficient because of the
following issues which could arise.

<center>
<img align="center" width=300 src="MLLD_blog/MLLD_blog/public/images/8c3f43d467a5739b3378d0aa9118910d.png">

Figure 2
</center>

Hence, we need to optimize the regular DL approach to be lightweight on the
memory and computation.

### **Solving the Deep Learning optimization problem through Deep Compression**

We approach this problem through a three-step solution:

1. Pruning

2. Weight Sharing (Trained Quantization)

3. Huffman Coding

In Figure 0, we saw the all the neurons were “interconnected” with each other. A
fully connected architecture requires the storage of the weights of all the
connections in the architecture.  
More the network is connected, the heavier it becomes, i.e. more data associated
with the network needs to be stored, computed and updated.

Hence the idea of Deep Compression is to reduce the heaviness of the Deep Neural
Networks helping the mobile devices access DL easily.

### **1. Network Pruning**

Deep neural networks try to mimic the way synaptic connections form between neurons in the human brain. A newborn baby constantly learns and makes new neural connections, comparable to training a deep neural network. However, it is interesting to note that the number of neural connections peak at the age of one and reduce and stabilize by the age of ten. This suggests that some of the connections formed at an early stage are pruned or removed naturally. In the same way, we can [prune deep neural networks](https://blog.paperspace.com/neural-network-pruning-explained/) to remove redundant weights and nodes. Pruning would compress the model and decrease the space and computational requirements, making the deployment of deep neural networks in real-world scenarios easier and achievable.
<p align="center">
<img src="MLLD_blog/MLLD_blog/public/images/synapses.png" alt="Kitten"
 title="A cute kitten" width="400" height="200" />
</p>

### **What to prune?**

One can perform neural network pruning in two ways, either by pruning weights or by making the network architecture smaller by removing nodes. Which weights and nodes should be removed can be decided based on various heuristics. One possible approach is to remove the weights that are close to zero or lower in magnitude. Apart from weights, even the activations on training data can be used to prune the neural network. The neurons that produce near-zero values while training can be removed from the model without hurting the model's accuracy. In case the model’s accuracy reduces while pruning, it is recovered using finetuning.
<center>
<img src="MLLD_blog/public/images/prune.png" alt="Kitten"
 title="A cute kitten" width="350" height="200" />
<img src="MLLD_blog/public/images/pruning_process.png" alt="Kitten"
 title="A cute kitten" width="200" height="200" />
</center>

</br>

## **2. Quantization**

[Quantization in Deep Learning](https://medium.com/@joel_34050/quantization-in-deep-learning-478417eab72b) approximates a neural network's floating-point numbers to low bit-width numbers. [Parameter Sharing (or weights replication)](https://towardsdatascience.com/understanding-parameter-sharing-or-weights-replication-within-convolutional-neural-networks-cc26db7b645a) is another simple method that limits the number of stored parameters. Although combining these concepts with the pruned network leads to a less precise model, it is fair trade as it compresses our model while maintaining accuracy.

Methodology to identify the quantized weights

Quantized weights of <em>n</em> original weights <em>W = {w1, w2, …, wn}</em> in a single layer into <em>k</em> clusters <em>C = {c1, c2, ..., ck}</em>, <em>n>>k</em> is described below. [K-Means Clustering](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1) identifies the shared quantized weights for each layer, <em>k</em> is dependent on the limit placed on the number of stored parameters.
<center>
<img src="MLLD_blog/public/images/image_1.JPG" width="400">

Figure 3: Structure of proposed deep compression technique- 1. Pruning and 2. Quantization (current step being discussed).
</center>

<br/>

### **1. Cluster the weights**

<img align="right" width=250 src="MLLD_blog/public/images/image3.JPG">

[**Initialization of k centroids**](http://sbubeck.com/BML_PS12.pdf) is a crucial step as it influences the network's prediction accuracy—initialization techniques discussed in the paper include - [Forgy(random)](https://jamesmccaffrey.wordpress.com/2020/04/16/k-means-clustering-forgy-initialization-vs-random-initialization/), Density-based, and [Linear](https://arxiv.org/pdf/1409.3854.pdf). 

During initialization, the larger the weight, the more significant its contribution, but these large weights are scarce, causing forgy and density-based initialization to poorly represent these few large weights. Linear initialization bypasses this as it is independent of the magnitude of the weight.

### **2. Generate Code Book**

>**Assign the clustered index to each weight**: The original n weights of the layer are assigned to one of the centroids by minimizing the within-cluster sum-of-squares.

<img align="center" width=250 src="MLLD_blog/public/images/image4.JPG">
<br/>

### **3. Quantize the weights with the Code Book**

>**Quantize weights into k clusters**: All the weights in the same cluster share the same value. As a result, we only store a small number of weights into a table of shared weights.


### **4. Retrain Code Book**

>**Calculate the gradient of the loss function** for each weight (k centroids): Sum the gradients of weights in each cluster, multiply this by the learning rate and subtract from the cluster centroids from the previous iteration.

<img align="center" width=250 src="MLLD_blog/public/images/image5.JPG">
<br/>

**Quantization Compression Rate**

>In the case of k clusters, we need **<em>log<sub>2</sub>k</em>** bits to encode it. In a neural network with n connections and each connection is represented with b bits, restraining the connections to have only k shared weights will result in a compression rate of,

<center>
<img align="center" width=250 src="MLLD_blog/public/images/image2.JPG">
Equation to calculated the gradient of the centroids, here L- Loss, Wi,j - weight ith column and jth row, the centroid index of element Wi,j by Iij , the kth centroid of the layer by Ck.
</center>

## **3. Huffman Coding**

>Huffman code is a particular type of optimal prefix code that is commonly used for lossless data compression. The process of finding or using such a code proceeds by means of [Huffman coding](https://en.wikipedia.org/wiki/Huffman_coding). Huffman coding produces a variable-length code table for encoding source symbols. The table is derived from the occurrence probability for each symbol. As in other entropy encoding methods, more common symbols are represented with fewer bits than less common symbols, thus saving the total space.

### **Experiments**

>The research paper examines four networks that were "compressed." We will discuss their results briefly. Table 1 shows the network parameters and Top-1 before and after pruning. We notice a tremendous decrease in the network storage by 35× to 49× across the networks with no loss of accuracy. From Table 2, we can identify that most of the saving comes from pruning and quantization, while Huffman coding gives a marginal gain.
<center>
<img align="center" src="MLLD_blog/public/images/image6.JPG">
<br/>
Table 1: The compression pipeline can save 35× to 49× parameter storage with no loss of accuracy.
</center>

<center>
<img align="center" src="MLLD_blog/public/images/image7.JPG">
<br/>
Table 2: Compression statistics for  LeNet-300-100, LeNet-5, AlexNet and VGG-16. P: pruning, Q:quantization, H:Huffman coding.
</center>

### **Deep Compression vs. Singular Value Decomposition**

>We could use SVD as a baseline to understand the effectiveness of Deep
Compression in saving the loss inaccuracy.  

>SVD is one of the most essential linear algebra concepts in Machine Learning.
It is a mathematical transformation of a matrix to represent that matrix in a
lower dimension.  
This, in turn, helps us reduce the dimensionality problem often faced in Data
Science.

>SVD states that any matrix A can be factorized as: U S VT
<center>
<img width=400 src="MLLD_blog/public/images/fdf62537b0bcabc9571b4c40bf372da4.png">

Figure 3
</center>
<br/>

>We have now been able to go from m\*n to m\*m, n\*n, and a sparse m\*n diagonal
matrix.
Which saves us space and reduces dimensionality.

**Enter Deep Compression.**

Deep Compression also is built to achieve similar results where we are trying to
compress the representation of the DL best model, which we have learned. We want
to squeeze the model to make it accessible to mobile devices with lighter
computation.
<center>
<img width=400 src="MLLD_blog/public/images/a5a46ac82dfbbb39fda4ebd118067a6a.png">

Figure 4
</center>

<br/>

Upon Compressing AlexNet through Deep Compression (DC) techniques and SVD, we
observe that,

- SVD has a meager compression ability as compared to DC techniques, which
    means our idea of DC is a success

- DC techniques also achieve a lesser accuracy loss as compared to SVD, which
    means DC is high performance and memory-efficient

- Finally, we are also able to see how an ensemble of Pruning & Quantization
    together are essential to achieve the best results inability to compress and
    ability to maintain original accuracy post-compression

Deep Compression is onto something big! It beats the traditional SVD. Hence, we
should try to implement it in real-world Deep Learning models than use them
directly.

### **SpeedUP**

>Many real world applications of deep learning like pedestrian detection require fast inference. The fully connected layers are the heaviest in most deep convolutional networks and these fully connected layers are getting compressed the most from deep compression. Hence deep compression is particularly beneficial for applications that focus on extremely low time latency.

<center>
<img src="MLLD_blog/public/images/speedup.png" alt="Kitten"
 title="A cute kitten" width="600" height="200" />
FigureXXX: Compared with the original network, pruned network layer achieved 3x speedup on CPU, 3.5x on GPU and 4.2x on mobile GPU on average.
</center>

>We notice that Deep Compression could cause a loss in accuracy due to pruning,
quantization, or encoding. Hence, we need to experimentally identify how much we
should compress our DL model instead of randomly picking a number.

# **Conclusion**

>A neural network deep compression technique that doesn’t affect the accuracy was discussed; it included pruning unimportant connections, quantizing the network by limiting the weights, using k-means, and applying Huffman coding.

>Results show compression by 35x to 49x without loss of accuracy. After Deep Compression, the size of these networks fit into the on-chip SRAM cache rather than requiring off-chip DRAM memory, making deep neural networks more energy efficient to run on mobile. The neural network can be pruned during or after training. It improves inference time/ model size vs. accuracy tradeoff for a given architecture and uses it in convolutional and fully connected layers. 

>However, it generally does not help as much as switching to better architecture. Quantization can also be applied both during and after training. It can be used for both convolutional and fully connected layers. But quantized weights make neural networks harder to converge, thus requiring a lower learning rate to ensure the network to have good performance. 


>Additionally, they make back-propagation infeasible since gradient cannot back-propagate through discrete neurons requiring approximation methods to estimate the gradients of the loss function concerning the input of the discrete neuron. These cons make us want to investigate other models compression techniques such as Knowledge distillation, Selective Attention, and Low-rank factorization and compare their results.