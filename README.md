# Neural-Net-Python
Neural Net classification package in Python:
Uses backpropagation and conjugate gradient optimization in order to solve for the weights of a neural net. 

Data example is taken from Coursera Machine Learning course


For ann use class MLPClassifier (similar to scikit_learn).
example:

    clf = MLPClassifier(lambda_reg=(1.0), hidden_layer_sizes=(250,), tol=0.005, cg_solver='fmin_cg')
    clf.fit(x_data, y_data)
    clf.predict(input_x)
        
options for defining MLPClassifier class:

    lambda_reg: [float] [default = 1e-05] 
        regularization parameter (called alpha in scikit_learn)
    hidden_layer_sizes: [tuple of integers] [default = (5, 2)]
        each element represents a hidden layer with the integer representing 
        the number of hiddin units in that layer
    tol: [float] [default = 0.0001]
        convergence criteria or tolerance for when to stop conjugate gradient. 
        If change in cost is below this value, the process terminates. 
    cg_solver: [string] [default = 'fmin_cg']
        Options for conjugate gradient methods are either 'fmin_cg' which 
        uses optimize.fmin_cg from the scipy package or 'nl_cg' which uses 
        built from scratch conj_grad package
        
methods:
    fit(x, y):
        trains the neural neight (solves for the weights) by 
        (1) random initialization of weights
        (2) conjugate gradiant optimization using backpropagation
        (3) outputs weights to file, 'weights.dat' and stores them in the 
            class 
        input:
            x: [numpy matrix]
                input data features
            y: [numpy matrix]
                input data classes
    read_weights(n, k):
        alternative option for determining weights to use. Here, weights are
        read in from a supplied file, "weights.dat"
        input:
            n: [integer]
                number of features
            k: [integer]
                number of classes
    predict(x, threshold):
        makes a prediction based on weights. 
        input:
            x: [numpy matrix]
                numpy matrix of dimension 1 x n, where n is the number of features
            threshold: [float]
                Value which indicates the cutoff for determining a true or false 
                designation in the prediction
        output:
            [numpy array]
                numpy array of length equal to the number of classes containing 
                1's and 0's (true or false)
    cost(x, y):
        returns cost function for given data
        inputs:
            x: [numpy matrix]
                input data features
            y: [numpy matrix]
                input data classes
        outputs:
            [float]
                cost fucntion value
                
                
Can be ran as standalone program with three run types passed in as integers 1, 2, or 
3 as arguments.
Requires 2 files to be present:
    training_x.dat: the data for x values
    training_y.dat: the data for corresponding y values 
  
run type 1: Train weights from training set data
    Uses all of the data in training_x.dat as the training set and uses the fit method 
    to determine weights
run type 2: Print diagnostic values for cross validation and test sets from prexisting 
    weights
    Uses 60 % of training_x.dat as the training set, 20 % as the cross-validation set, 
    and 20% as the test set
run type 3: Print diagnostic curve for regularization parameter lambda
    Prints and graphs the minimized cost after training as a function of the 
    regularization parameter, lambda for each of the three sets. This functionality 
    is used in order to determine the optimal value for lambda. 
        
        
    
