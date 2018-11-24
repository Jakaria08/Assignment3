from __future__ import division  # floating point division
import numpy as np
import utilities as utils

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it should ignore this last feature
        self.params = {'usecolumnones': True}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.means = []
        self.stds = []
        self.numfeatures = 0
        self.numclasses = 2
        self.prior_prob = []

    def learn(self, Xtrain, ytrain):
        """
        In the first code block, you should set self.numclasses and
        self.numfeatures correctly based on the inputs and the given parameters
        (use the column of ones or not).

        In the second code block, you should compute the parameters for each
        feature. In this case, they're mean and std for Gaussian distribution.
        """

        ### YOUR CODE HERE
        if self.params['usecolumnones']:
            self.numfeatures = Xtrain.shape[1]
            
        else:
             self.numfeatures = Xtrain.shape[1] - 1

        ### END YOUR CODE

        origin_shape = (self.numclasses, self.numfeatures)
        self.means = np.zeros(origin_shape)
        self.stds = np.zeros(origin_shape)
        self.prior_prob = np.zeros(2)
    

        ### YOUR CODE HERE
        for clas in range(self.numclasses):
            indices = np.where(ytrain == clas)
            trainclass = Xtrain[indices]
            ## calculating prior probability for each class
            self.prior_prob[clas] = np.size(indices)/(float)(Xtrain.shape[0])
            #print(np.size(indices))
            for i in range(self.numfeatures):
                self.means[clas, i] = np.mean(trainclass[:, i])
                self.stds[clas, i] = np.std(trainclass[:, i])
            
        ### END YOUR CODE

        assert self.means.shape == origin_shape
        assert self.stds.shape == origin_shape

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE
        Xtest = Xtest[:,0:self.numfeatures]
        
        # calculate likelihood
        for i in range(Xtest.shape[0]):
            probabilites = np.ones(self.numclasses)
            for k in range(self.numclasses):
                for j in range(Xtest.shape[1]):
                    if j == Xtest.shape[1] - 1 and self.params['usecolumnones'] == True:
                        continue
                    probabilites[k] *= ((1 / (np.sqrt(2*np.pi * np.power(self.stds[k,j],2))))
                    * np.exp(-(np.power((Xtest[i,j]-self.means[k,j]),2)/(2*np.power(self.stds[k,j],2)))))
                #marginal probability
            probabilites *= self.prior_prob
            
            if(probabilites[0]>probabilites[1]):
                ytest[i] = 0
            else:
                ytest[i] = 1
              
            
            
        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest

class LogitReg(Classifier):

    def __init__(self, parameters={'regularizer': 'l2'}):
        self.params = parameters
        self.iters = 1000
        self.learning_rate = .001
        self.numclasses = 2
        self.reg_weight = 0.0
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        self.bias = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))


    def logit_cost(self, theta, X, y):
        """
        Compute cost for logistic regression using theta as the parameters.
        """
        numsamples = X.shape[0]
        cost = 0.0

        ### YOUR CODE HERE
        
        h = utils.sigmoid(np.dot(X, theta))
        
        if 'regwgt' in self.params:
            reg = (self.params['regwgt'] / (2 * numsamples)) * np.sum(theta**2)
            cost = (1 / numsamples) * (np.dot(-y.T,(np.log(h))) - np.dot((1 - y).T,(np.log(1 - h)))) + reg
        else:
            cost = (1 / numsamples) * (np.dot(-y.T,(np.log(h))) - np.dot((1 - y).T,(np.log(1 - h))))
            
        ### END YOUR CODE

        return cost

    def logit_cost_grad(self, theta, X, y):
        """
        Compute gradients of the cost with respect to theta.
        """
        grad = np.zeros(len(theta))
        ### YOUR CODE HERE
    
        h = utils.sigmoid(np.dot(X, theta)+self.bias)
        
        if 'regwgt' in self.params:
            reg = (self.params['regwgt'] / 2) * theta
            grad =  np.dot(X.T,(h - y)) + reg
        else:
            grad =  np.dot(X.T,(h - y))
            
    
        ### END YOUR CODE
        
        return grad
    
    def logit_cost_bias(self, theta, X, y, bias):
        """
        Compute gradients of the cost with respect to theta.
        """
        numsamples = X.shape[0]
        h = utils.sigmoid(np.dot(X, theta)+self.bias)
        self.bias = (1/numsamples) * np.sum(h - y) 
    
        return self.bias



    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data
        """

        self.weights = np.zeros(Xtrain.shape[1],)
        self.bias = 0
        
        ### YOUR CODE HERE
        
        for step in range(self.iters):
            self.weights -= self.learning_rate * self.logit_cost_grad(self.weights, Xtrain, ytrain)
            self.bias -= self.learning_rate * self.logit_cost_bias(self.weights, Xtrain, ytrain, self.bias)
            new_cost = self.logit_cost(self.weights, Xtrain, ytrain)
            
            #if step % 100 == 0:
                #print("Logistic Regression Cost: " + str(new_cost))
                #print("Logistic Regression bias: " + str(self.logit_cost_bias(self.weights, Xtrain, ytrain, self.bias)))
                
        ### END YOUR CODE

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE
        prediction = utils.sigmoid(np.dot(Xtest, self.weights))
        
        for i in range(len(prediction)):
            if prediction[i] >= 0.5:
                ytest[i] = 1
                
        ### END YOUR CODE
        assert len(ytest) == Xtest.shape[0]
        return ytest

class NeuralNet(Classifier):
    """ Implement a neural network with a single hidden layer. Cross entropy is
    used as the cost function.

    Parameters:
    nh -- number of hidden units
    transfer -- transfer function, in this case, sigmoid
    stepsize -- stepsize for gradient descent
    epochs -- learning epochs

    Note:
    1) feedforword will be useful! Make sure it can run properly.
    2) Implement the back-propagation algorithm with one layer in ``backprop`` without
    any other technique or trick or regularization. However, you can implement
    whatever you want outside ``backprob``.
    3) Set the best params you find as the default params. The performance with
    the default params will affect the points you get.
    """
    def __init__(self, parameters={}):
        self.params = {'nh': 16,
                    'transfer': 'sigmoid',
                    'stepsize': 0.01,
                    'epochs': 1000}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
        self.w_input = None
        self.w_output = None

    def feedforward(self, inputs):
        """
        Returns the output of the current neural network for the given input
        """
        # hidden activations 5000*16
        a_hidden = self.transfer(np.dot(inputs, self.w_input))
        
        # output activations 5000*1
        a_output = self.transfer(np.dot(a_hidden, self.w_output))

        return (a_hidden, a_output)

    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_input, nabla_output)`` representing the gradients
        for the cost function with respect to self.w_input and self.w_output.
        """

        ### YOUR CODE HERE
        #forward pass
        y = y.reshape(-1,1)
        a_hidden_out, a_output_out = self.feedforward(x)
        # backward propagate through the network
        output_layer_error = a_output_out - y # error in output
        
        output_delta = np.multiply(output_layer_error, self.dtransfer(a_output_out)) #5000*1

        hidden_layer_error = np.dot(output_delta, np.transpose(self.w_output)) #5000*16
        hidden_delta = np.multiply(hidden_layer_error, self.dtransfer(a_hidden_out)) #5000*16 
        
        #print(np.shape(self.w_input))
        
        delta_hidden_layer = np.dot(x.T, hidden_delta) #9*16
        delta_output_layer = np.dot(a_hidden_out.T, output_delta ) #16*1
        
        nabla_input = delta_hidden_layer
        nabla_output = delta_output_layer
        ### END YOUR CODE

        assert nabla_input.shape == self.w_input.shape
        assert nabla_output.shape == self.w_output.shape
        return (nabla_input, nabla_output)

    # TODO: implement learn and predict functions
    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data
        """
        # Weight initialization
        self.w_input = np.random.randn(Xtrain.shape[1], self.params['nh'],) 
        self.w_output = np.random.randn(self.params['nh'], 1)
        
        for iter in range(self.params['epochs']):
            shuffle = np.arange(Xtrain.shape[0])
            np.random.shuffle(shuffle)
            Xtrain = Xtrain[shuffle, :]
            ytrain = ytrain[shuffle]
            nabla_input, nabla_output = self.backprop(Xtrain, ytrain)
            self.w_output = self.w_output - self.params['stepsize'] * nabla_output
            self.w_input = self.w_input - self.params['stepsize'] * nabla_input
            hidden, output = self.feedforward(Xtrain)
            #print("Loss: \n" + str(np.mean(np.square(ytrain - output ))))
                
    def predict(self, Xtest):
        numsamples = Xtest.shape[0]
        ytest = np.zeros(numsamples)
        for i in range(numsamples):
            if self.feedforward(np.transpose(Xtest[i, :]))[1] >= 0.5:
                ytest[i] = 1
            else:
                ytest[i] = 0
        return ytest

class KernelLogitReg(LogitReg):
    """ Implement kernel logistic regression.

    This class should be quite similar to class LogitReg except one more parameter
    'kernel'. You should use this parameter to decide which kernel to use (None,
    linear or hamming).

    Note:
    1) Please use 'linear' and 'hamming' as the input of the paramteter
    'kernel'. For example, you can create a logistic regression classifier with
    linear kerenl with "KernelLogitReg({'kernel': 'linear'})".
    2) Please don't introduce any randomness when computing the kernel representation.
    """
    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None', 'kernel': 'linear'}
        self.num_of_center = 50
        self.iteration = 10000
        self.learning_rate = .1
        self.n = None
        self.reset(parameters)
        
    def kernel_logit_cost(self, theta, X, y):
        """
        Compute cost for logistic regression using theta as the parameters.
        """
        numsamples = X.shape[0]
        cost = 0.0

        ### YOUR CODE HERE
        
        h = utils.sigmoid(np.dot(X, theta))
        
        if 'regwgt' in self.params:
            reg = (self.params['regwgt'] / (2 * numsamples)) * np.sum(theta**2)
            cost = (1 / numsamples) * (np.dot(-y.T,(np.log(h))) - np.dot((1 - y).T,(np.log(1 - h)))) + reg
        else:
            cost = (1 / numsamples) * (np.dot(-y.T,(np.log(h))) - np.dot((1 - y).T,(np.log(1 - h))))
            
        ### END YOUR CODE

        return cost

    def kernel_logit_cost_grad(self, theta, X, y):
        """
        Compute gradients of the cost with respect to theta.
        """

        h = utils.sigmoid(np.dot(X, theta))
        
        if 'regwgt' in self.params:
            reg = (self.params['regwgt'] / 2) * theta
            grad =  np.dot(X.T,(h - y)) + reg
        else:
            grad =  np.dot(X.T,(h - y))
        
    
        ### END YOUR CODE
        
        return grad
        
    def kernel(self, X, N):
        if self.params['kernel'] == 'linear':
            return np.dot(X, N.T)
        elif self.params['kernel'] == 'hamming':
            Ktrain = np.zeros((X.shape[0], N.shape[0]))
            for i in range(X.shape[0]):
                hamming_distance = np.zeros(N.shape[0])
                for j in range(N.shape[0]):
                    if type(X[i]) is str:
                        y = 0
                        z = X[i] ^ N[j]
                        while z:
                            y += 1
                            z &= z - 1
                        hamming_distance[j] = y
                    else:
                        hamming_distance[j] = 0 if X[i] == N[j] else 1
                Ktrain[i] = hamming_distance
            return Ktrain
        else:
            return X;

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data.

        Ktrain the is the kernel representation of the Xtrain.
        """
        Ktrain = None

        ### YOUR CODE HERE
        #Creating centers and sending to kernel
        
        self.n = Xtrain[0:self.num_of_center]
        Ktrain = self.kernel(Xtrain, self.n)
        ### END YOUR CODE

        self.weights = np.zeros(Ktrain.shape[1],)

        ### YOUR CODE HERE
        for i in range(self.iteration):
            self.weights -= (self.learning_rate / (i+1)) * self.kernel_logit_cost_grad(self.weights, Ktrain, ytrain)
            #print("Loss: "+str(self.kernel_logit_cost(self.weights, Ktrain, ytrain)))
        ### END YOUR CODE

        self.transformed = Ktrain # Don't delete this line. It's for evaluation.

    # TODO: implement necessary functions
    def predict(self, Xtest):
        Ktest = self.kernel(Xtest, self.n)
        ytest = np.zeros(Ktest.shape[0], dtype=int)
        prediction = utils.sigmoid(np.dot(Ktest, self.weights))
        for i in range(len(prediction)):
            if prediction[i] >= 0.5:
                ytest[i] = 1

        return ytest


# ======================================================================

def test_lr():
    print("Basic test for logistic regression...")
    clf = LogitReg()
    theta = np.array([0.])
    X = np.array([[1.]])
    y = np.array([0])

    try:
        cost = clf.logit_cost(theta, X, y)
    except:
        raise AssertionError("Incorrect input format for logit_cost!")
    assert isinstance(cost, float), "logit_cost should return a float!"

    try:
        grad = clf.logit_cost_grad(theta, X, y)
    except:
        raise AssertionError("Incorrect input format for logit_cost_grad!")
    assert isinstance(grad, np.ndarray), "logit_cost_grad should return a numpy array!"

    print("Test passed!")
    print("-" * 50)

def test_nn():
    print("Basic test for neural network...")
    clf = NeuralNet()
    X = np.array([[1., 2.], [2., 1.]])
    y = np.array([0, 1])
    clf.learn(X, y)

    assert isinstance(clf.w_input, np.ndarray), "w_input should be a numpy array!"
    assert isinstance(clf.w_output, np.ndarray), "w_output should be a numpy array!"

    try:
        res = clf.feedforward(X[0, :])
    except:
        raise AssertionError("feedforward doesn't work!")

    try:
        res = clf.backprop(X[0, :], y[0])
    except:
        raise AssertionError("backprob doesn't work!")

    print("Test passed!")
    print("-" * 50)

def main():
    test_lr()
    test_nn()

if __name__ == "__main__":
    main()
