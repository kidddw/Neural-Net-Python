#!/usr/bin/python -tt

#add other potentials


"""Neural Net learning algorithm
   one-vs-all classification   

   -Import input data
   -Import number of layers and features per layer from file
   -Initialize weights
   -Perform backpropagation with conjugate gradient to optimize weights
   -Output weights

   if "training" option is turned on, use backpropagation to learn weights and 
   write to file
   if "training" option is turned off, use preexisting weights from file to 
   assess error

   To Do:
     - impliment conjugate gradient
     - create architecture class which contains m, L, b_reg
"""

import sys
import numpy as np
from scipy.linalg import eigh
from scipy import optimize
import matplotlib.pyplot as plt
import random
import conj_grad as cg


class Define_Data:

    def __init__(self, x_fname, y_fname, perc):
        if perc[0] + perc[1] + perc[2] != 1.0: 
            print 'percentage split of data must equal 1.0'
            sys.exit()
        X_tot, Y_tot = self.import_data(x_fname, y_fname)
        total_n = X_tot.shape[0]
        train_n = int(perc[0] * total_n)
        cv_n = int((perc[0] + perc[1]) * total_n)
        self.X_train = X_tot[:train_n, :]
        self.Y_train = Y_tot[:train_n, :]
        self.m_train = self.X_train.shape[0]
        self.X_CV = X_tot[train_n:cv_n, :]
        self.Y_CV = Y_tot[train_n:cv_n, :]
        self.m_CV = self.X_CV.shape[0]
        self.X_test = X_tot[cv_n: ,:]
        self.Y_test = Y_tot[cv_n:,:]
        self.m_test = self.X_test.shape[0]

    def import_data(self, x_fname, y_fname):
        """
           Import data. X matrix as numpy matrix type, y vector as array type
           Assume y data is presented as integers ranging from 1 to 
           number-of-classes
        """ 

        f=open(x_fname, 'rU')
        lines = f.read().splitlines()
        order = (np.arange(len(lines)))
        np.random.shuffle(order)
        arr = [] 
        for i in range(len(lines)):
            row = []
            for num in lines[order[i]].split():
                row.append(float(num))
            arr.append(row)
        Xmat = np.matrix(arr)
        f.close()

        f=open(y_fname, 'rU')
        lines = f.read().splitlines()
        yvec = []
        for i in range(len(lines)):
            num = float(lines[order[i]].split()[0])
            yvec.append(int(num))
        f.close()

        if len(yvec) != Xmat.shape[0]: 
            print 'data length mismatch'
            sys.exit()
        m = len(yvec)
        K = len(set(yvec))
        Ymat = np.matrix(np.zeros((m, K)))
        for i in range(m):
            Ymat[i, (yvec[i] - 1)] = 1
        return Xmat, Ymat


def import_architecture(fname):
    """Import neural network archatecture. amount of units in hidden layers""" 
    f=open(fname, 'rU')
    lines = f.read().splitlines()
    arch_vec = []
    for line in lines:
        if line[0] == '#': continue
        arch_vec.append(int(line))
    return arch_vec


def import_weights(arch, file_type):
    """read in Theta matrices from ascii file""" 
    tens = []
    if file_type == 1:
        for l in range(len(arch) - 1):
            fname = 'Theta' + str(l + 1) + '.dat'
            f = open(fname, 'rU')
            lines = f.read().splitlines()
            arr = []
            for line in lines:
                row = []
                for num in line.split():
                    row.append(float(num))
                arr.append(row)
            mat = np.matrix(arr)
            f.close()
            tens.append(mat)
    if file_type == 2:
        f = open('weights.dat', 'rU')
        lines = f.read().splitlines()
        ind = 1
        for l in range(len(arch) - 1):
            arr = []
            for i in range(arch[l + 1]):
                row = []
                for j in range(arch[l] + 1):
                    row.append(float(lines[ind]))
                    ind = ind + 1
                arr.append(row)
            mat = np.matrix(arr)
            tens.append(mat)
        f.close()
    return tens


def initialize_theta(arch, epsilon):
    """initialize theta matrices with random numbers between -epsilon and 
       epsilon. list of numpy matrices"""
    tens = []
    for l in range(len(arch)):
        if l == 0: continue
        mat = []
        n = arch[l]
        m = arch[l - 1] + 1
        for i in range(n):
            row = []
            for j in range(m):
                num = random_num(epsilon)
                row.append(num)
            mat.append(row)
        tens.append(np.matrix(mat))
    return tens


def random_num(eps):
    """generate a random number between the range of -eps and eps"""
    num = random.uniform(-eps, eps)
    return num


def sigmoid(mat):
    """replace each element with the sigmoid function evaluated using that 
       element's value"""
    g = np.divide(1.0, np.add(1.0, np.exp(-mat)))
    return g


def sigmoid_grad(mat):
    """replace each element with the sigmoid function evaluated using that 
       element's value"""
    g = np.multiply(sigmoid(mat), np.subtract(1.0, sigmoid(mat)))
    return g


def add_one(mat):
    """concatinate a column of ones onto the leftmost side of the matrix"""
    m = mat.shape[0]
    mat_ones = np.matrix(np.concatenate((np.ones((m,1)), mat), axis = 1))
    return mat_ones


def cost_function(Theta_mat, data, lambd):
    """calculate the cost function for given theta matrices and lambda.
       Note: can be faster if not using trace"""
    b_reg = []  # Matrices for regularization inclusion
    L = len(Theta_mat) + 1
    for l in range(L):
        if l == 0: continue
        Theta = Theta_mat[l - 1]
        b_reg.append(np.identity(Theta.shape[1]))
        b_reg[l - 1][0, 0] = 0.0
    hyp = feed_forward(Theta_mat, data.X_train)[L - 1]
    J = (1.0 / data.m_train) * np.trace(-np.dot(np.transpose(data.Y_train), 
        np.log(hyp)) - np.dot(np.transpose(np.ones(data.Y_train.shape) - 
        data.Y_train), np.log(np.ones(hyp.shape) - hyp)))
    for l in range(L - 1):
        J = J + ((0.5 * lambd / data.m_train) *
            (np.trace(np.transpose(Theta_mat[l]) * Theta_mat[l] * b_reg[l]))
    return J


def cost_function_vec(Theta_mat, data, lambd, arch):
    J = cost_function(rollup(Theta_mat, arch), data, lambd)
    with open('cost.dat', 'a') as file:
        file.write(str(J) + '\n')
    output_results(rollup(Theta_mat, arch), data.X_train, data.Y_train)
    return J


def feed_forward(Theta_mat, X):
    """calculate cost function for given theta matrices and lambda"""
    a = []
    z = []
    L = len(Theta_mat) + 1
    for l in range(L):
        if l == 0:  
            a.append(X) 
            z.append([0.0])
            continue
        Theta = Theta_mat[l - 1]
        a_vec = add_one(a[l - 1])
        z_vec = a_vec * np.transpose(Theta) # numpy matrix type
        z.append(np.array(z_vec))
        a.append(np.array(sigmoid(z_vec)))
    return a


def grad_cost_function(Theta_mat,data,lambd):
    """calculate gradient of cost function with respect to Theta matrices for 
       given theta matrices and lambda via backpropagation"""
    L = len(Theta_mat) + 1
    delta = [None] * L
    Delta = [None] * L
    # Feed forward training examples
    a = feed_forward(Theta_mat, data.X_train)
    # backprop to define delta^(l) for hidden layers
    for ll in range(L):
        if ll == L - 1: continue
        l = L - ll - 1
        if l == L - 1: 
            delta[l] = a[l] - data.Y_train
            continue
        A = delta[l + 1] * Theta_mat[l]
        B = sigmoid_grad(add_one(a[l - 1]) * np.transpose(Theta_mat[l - 1]))
        delta[l] = np.multiply(A, add_one(B))
        delta[l] = np.delete(delta[l], [0], axis = 1)
    b_reg = []  # Matrices for regularization inclusion
    Theta_grad = []
    for l in range(L):
        if l == L - 1: continue
        Delta[l] = np.transpose(delta[l + 1]) * add_one(a[l]) 
        Theta = Theta_mat[l]
        b_reg.append(np.identity(Theta.shape[1]))
        b_reg[l][0, 0] = 0.0
        Theta_grad.append((1.0 / data.m_train) * Delta[l] + 
                          (lambd / data.m_train) * Theta[l] * b_reg[l]) 
    return Theta_grad


def grad_cost_function_vec(Theta_mat, data, lambd, arch):
    return unroll(grad_cost_function(rollup(Theta_mat, arch), data, lambd), arch)


def grad_cost_function_fd(Theta_mat, data, lambd):
    """estimate gradient of cost function with respect to Theta matrices for 
       given theta matrices and lambda via finite difference"""
    L = len(Theta_mat) + 1
    e = 0.0001
    numgrad_mat = []
    Theta_mat_temp = Theta_mat
    for l in range(L):
        if l == L - 1: continue
        Theta = Theta_mat[l]
        numgrad = np.zeros(Theta.shape)
        perturb = np.zeros(Theta.shape)
        for i in range(Theta.shape[0]):
            for j in range(Theta.shape[1]):
                perturb[i, j] = e 
                Theta_mat_temp[l] = Theta - perturb 
                loss1 = cost_function(Theta_mat_temp, data, lambd)
                Theta_mat_temp[l] = Theta + perturb 
                loss2 = cost_function(Theta_mat_temp, data, lambd)
                numgrad[i,j] = (loss2 - loss1) / (2 * e) 
                perturb[i,j] = 0
        numgrad_mat.append(numgrad)
    return numgrad_mat
  

def unroll(matrices,arch):
  """return numpy matrix type column vector of unrolled matrixes. arch 
     determines the dimensions of each matrix"""
  L = len(arch)
  vec = []
  for l in range(L):
      if l == 0: continue
      n = arch[l]
      m = arch[l - 1] + 1
      for i in range(n):
          for j in range(m):
              vec.append(matrices[l - 1][i, j])
  return np.array(vec)


def rollup(vector,arch):
    """return list of numpy matrix type matrices from vector. arch determines 
       the dimensions of each matrix"""
    L=len(arch)
    tens = []
    ind = 0
    for l in range(len(arch)):
        if l == 0: continue
        mat = []
        n = arch[l]
        m = arch[l - 1] + 1
        for i in range(n):
            row = []
            for j in range(m):
                num = vector[ind]
                row.append(num)
                ind = ind + 1
            mat.append(row)
        tens.append(np.matrix(mat))
    return tens


def output_results(Theta_mat, X, Y):
    L = len(Theta_mat) + 1
    # Feed forward training examples
    res = np.matrix(feed_forward(Theta_mat, X)[L - 1])
    maxvals = res.max(1)
    i = 0
    for row in res:
        for j in range(res.shape[1]):
            if res[i, j] == maxvals[i]: res[i, j] = 1
            else: res[i, j] = 0
        i = i + 1
    success_count = np.sum(np.all(np.equal(res, Y), axis = 1))
    print 'success percentage: ', float(success_count) / Y.shape[0] * 100
    return (100 - float(success_count) / Y.shape[0] * 100)


def output_weights(Theta_mat, arch):
    LL = len(Theta_mat)
    theta_out = open('weights.dat', 'w')
    theta_out.write('#' + str(arch) + '\n')
    for l in range(LL):
        for i in range(Theta_mat[l].shape[0]):
            for j in range(Theta_mat[l].shape[1]):
                theta_out.write(str(Theta_mat[l][i, j]) + '\n')
        print Theta_mat[l]
    theta_out.close()


def main():

    if len(sys.argv) != 2:
        print 'usage: ./neural_net.py <run type>'
        print 'run type 1: Train weights from data'
        print 'run type 2: Print success percentage from prexisting weights'
        sys.exit(1)
    run_type = int(sys.argv[1])

    perc = [0.6, 0.2, 0.2]
    mydata = Define_Data('training_x.dat', 'training_y.dat', perc)
    arch = ([mydata.X_train.shape[1]] + import_architecture('arch.dat') + 
            [mydata.Y_train.shape[1]])
    lambda_reg = 1.0


    # Train weights from training set data
    if run_type == 1:
        open('cost.dat', 'w').close()
        Theta_matrices = initialize_theta(arch,1.2)
        args = (mydata, lambda_reg, arch)
        # scipy conjugate gradient
        Theta_matrices = rollup(optimize.fmin_cg(cost_function_vec, 
            unroll(Theta_matrices,arch), fprime = grad_cost_function_vec, 
            args = args), arch)
        # home made conjugate gradient
    #    Theta_matrices = rollup(cg.nonlinear_cg(cost_function_vec, 
    #        unroll(Theta_matrices, arch), grad_cost_function_vec, args), arch)
        print
        print 'Training set'
        output_results(Theta_matrices, mydata.X_train, mydata.Y_train)
        print 'Cross-Validation set'
        output_results(Theta_matrices, mydata.X_CV, mydata.Y_CV)
        print 'Test set'
        output_results(Theta_matrices, mydata.X_test, mydata.Y_test)
        output_weights(Theta_matrices, arch)


    # Print diagnostic values for cross validation and test sets from 
    # prexisting weights
    if run_type == 2:
        Theta_matrices = import_weights(arch, 2)
        print 'Training set'
        output_results(Theta_matrices, mydata.X_train, mydata.Y_train)
        print 'Cross-Validation set'
        output_results(Theta_matrices, mydata.X_CV, mydata.Y_CV)
        print 'Test set'
        output_results(Theta_matrices,mydata.X_test, mydata.Y_test)


    # Print diagnostic curve for regularization parameter lambda
    if run_type == 3: 
        lambd_n = 10
        lambd_a = -4.0
        lambd_b = 2.0
        lambda_arr = []
        train_curve = [] 
        CV_curve = []
        test_curve = []
        for i in range(lambd_n):
            expon = lambd_a + (lambd_b - lambd_a) / (lambd_n - 1) * i
            lambda_reg = 10 ** expon
            lambda_arr.append(expon)
            Theta_matrices = initialize_theta(arch, 1.2)
            args = (mydata, lambda_reg, arch)
            # scipy conjugate gradient
            Theta_matrices = rollup(optimize.fmin_cg(cost_function_vec, 
                unroll(Theta_matrices, arch), fprime = grad_cost_function_vec, 
                args = args), arch)
            # home made conjugate gradient
      #      Theta_matrices = rollup(cg.nonlinear_cg(cost_function_vec, 
      #          unroll(Theta_matrices, arch), grad_cost_function_vec, 
      #          args), arch)
            train_curve.append(output_results(Theta_matrices, mydata.X_train, 
                               mydata.Y_train))
            CV_curve.append(output_results(Theta_matrices, mydata.X_CV, 
                            mydata.Y_CV))
            test_curve.append(output_results(Theta_matrices, mydata.X_test, 
                              mydata.Y_test))
        reg_out = open('regularization.dat', 'w')
        for i in range(lambd_n): 
            reg_out.write("{:10.6f}".format(lambda_arr[i]) + ' ' + 
                "{:10.6f}".format(train_curve[i]) + ' ' + 
                "{:10.6f}".format(CV_curve[i]) + ' ' + 
                "{:10.6f}".format(test_curve[i]) + '\n')
        plt.plot(lambda_arr, train_curve, label="Train")
        plt.plot(lambda_arr, CV_curve, label="Cross-Valid.")
        plt.legend(bbox_to_anchor = (0., 1.02, 1., .102), loc = 3, ncol = 2, 
            mode="expand", borderaxespad = 0.)
        plt.xlabel('log($\lambda$)')
        plt.ylabel('error')
        plt.draw()
        plt.show()
        









if __name__ == '__main__':
  main()
