# DeepLearning-TensorFlow-Python

# What is Artificial Intelligence?
> Making a machine do the tasks that usually requires human intelligence to be solved.
  NLP, Robotics, Self-Driving vehicles, Expert Systems are some domains of AI.
	

# What is the relation between AI, ML and DL?
> AI is a bigger domain which is related with solving real life problems. Machine learning
  like a brain for AI, it deals with the learning process and then making decisions. In ML model learns from the data. Within ML,
  Deep learning is further subset of ML. DL is a an architecture which replicates the human brain with a neural network for
  decision making. 


# Machine Learning types
> Supervised: Such a learning in which you have output labels to tune the model. 
  Examples are: 1) Classification: You have data your goal is to put an observation into a group. e.g cats and dogs classification. 
  	            2) Regression: You have to predict a real value based on some historical data. e.g predicting house prices.
> Unsupervised: Such a learning we only have input features not output labels. 
  Examples are: 1) Clustering: You have to find some similarties in data and make groups and relations based on those similarities.
	              2) Association rule: rule-based machine learning method for discovering interesting relations between variables in large                    databases. It is intended to identify strong rules discovered in databases using some measures of interestingness
> Reinforcement: Such a learning in which agent learns from environment. It follows reward and penalties method.
  Examples are: Making a dog follow your command and reward him if it follows. If not, penalize the dog.
  Deep Q is an example of Reinforcement learning Algorithm.


# What is Deep Learning?
> Deep Learning is a branch of Machine Learning absed on a set of algorithms that attempt to model high-level abstractions in the data     by using a deep graph with multiple processing layers. It is composed of multiple linear and non linear transformations.
  Examples are: 1)Automatic Machine Translation.
		2)Automatic Handwriting Generation.
		3)Character Text Generation.
		4)Automatic Game Playing.
		5)Object Classification in Images.


# What is Cost in Machine Learning?
> Cost is the measure of difference between the actual value and the predicted values is called Cost.


# What is Optimization in Machine Learning?
> Optimization is the process of updating the values of parameters so that the cost is lesser and predcted values are closer to the actual values.


# What is Activation Function?
> Activation functions are kind of mapping functions that translates the input into outputs. It uses a threshold to produce an output.
  Examples are: 
  1) Linear: f(x) = x --> Used when we want to solve linear regression problem where we predict numerical value.
  2) Unit Step: f(x) = 0 if 0>x, 1 if x>= 0 -->  Used where we have to set some threshold like in binary classification problems.
  3) Sigmoid: f(x) = 1/1+e^-x --> Used when we want to map values to a value in the range 0 to 1.
  4) Tanh: f(x) = tanh(x) --> Used when we want to map valuein the range -1 and 1
  5) ReLU: f(x) = 0 if x<0, x if x>=0 -->Used when we want to map input values in the range of x and max(0,x). It maps -ve value to 0 
  6) Softmax: Used when there are more than 2 classes it gives distribution for each class.


# What are Tensors?
> Tensors are the standard way of representing data in Deep Learning. These are multi-dimensional array so these numbers are represented   in higher multi-dimensional arrays. Rank 0 is scalar. Rank represent the dimensionality of things. Rank 1 is vector. Rank 2 is called   Matrices. Rank 3 is called Tensors.
  In TensorFlow the data has to be converted into tensors before any kind of calculations.


# What is TensorFlow?
> TensorFlow is a Python based open source library for Machine Learning to implement deep networks. In TensorFlow, computation is         reffered as dataflow graph.
  There are two sections in TensorFlow 1) Tensors 2) Flow: flow is the definition of graph like calculation and running the graph.
  In TensorFlow there are three types of data objects:
  1) Constants: Constant is a value that dosen't change. float32 is default for constants. 
  2) Placeholders: Placeholder is a promise to provide a value later. Most of the features and labels would be initiated as placeholder.   Placeholder comes with feed_dict which would feed final values in form of dicitonary.
  3) Variables: Variables allows us trainable parameter to the graph. These values can be changed during the running of the program.

# What is a perceptron?
> A perceptron is a single layered, single neuron which takes some input and produces an output. A perceptron is a linear model used for   binary classification.
  It has two functions 1) Summation and 2) Transformation(Activation).

# What is Role of weights and biases?
> Weights determine the slope of the classifier line that classifies data while bias is to shift the line forwad or backward.

# What is Learning rate and Epochs? 
> Learning rate is the amount each weight is corrected each time it is updated. The number  of time to run through the training data while updating weight.

# What is loss function?
> A loss function measures how far apart the current model is from the provided data. It measures how much the error is. It tells how     good or bad the model is.

# What is Back propagation?
> Back propagation allows us to find deltas for hidden layers. It allows us to vary parameters. Tracing the contribution of each unit     hidden or not.  
  Backpropagation of errors is an algorithm for neural networks using gradient descent. It consists of calculating the contribution of     each parameter to the errors.
  We backpropagate the errors through the net and update the parameters (weights and biases) accordingly.

# Note: The dataset is divided into three different parts. 1) Training data 2) Validation data 3) Test data
      But for this data set must be large.
      What happens if data set is very small. If you divide that data set into 3 parts then there may be a chance that the model won't         train properly.
      Solution: Solution to this problem is "N-fold Cross-Validation" - Which is a strategy which combines training and validation. 

# What is Overfitting?
> Overfitting is problem that our training set has focused on the particular training set so much, it has missed the point. 
  Generalization is poor.
  Random noise is captured too.

# How to spot over-fitting? 
> Validation data is what helps us to see if there is over-fitting. At some point validation loss could start increasing than the         training loss. 
  At this point you should stop the training of model. It is important that the model should not train on validation data.
  
# What is Early Stopping?
> Is a technique to prevent over-fitting. Common ways are 1) Train for preset number of epchs. But not guarantee the min is reached 2)     Stop when loss function updates become very small. But can lead to over-fitting. 3) Third is Validation test strategy. But may take     long time.

# What is Underfitting?
> Underfitting is problem that the model has not captured the underlying logic of the data. It doesn't know what to do and gives wrong     answer.    
  High loss and low accuracy.
  Test data is data which is used to measure predictive power after training and validation. The  accuracy of model on this data would     equal to real life data.
 
# Note: Well-trained model is somewhere between underfitted and overfitted model. This fine balance is called Bias-Variance dillema.

# What is Standardization/ Feature scalling?
> Standardization/ Feature scalling is the process of transforming the data we are working with into a standard scale e.g subtracting     the mean from original value and dividing by 
  standard deviation. In this way regardless of data set we'll always obtain a distribution with mean 0 and standard deviation 1. Other   popular methods are L2-norm, PCA or whitening

# How to encode categories in a useful way for ML?
> There are two ways:
  1) One hot encoding:
     Create values according to how many products. Problem is it requires a lot of variables.     
  2) Binary encoding:
     Convert all categories in binary values. But can be problematic because of correlations.



