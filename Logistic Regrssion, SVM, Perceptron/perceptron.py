# ---------------------------------------------#
# -------| Written By: Sibt ul Hussain |-------#
# ---------------------------------------------#

# A Perceptron algorithm with regularized weights...

from classifier import *
#from optimizer import * 
#Note: Here the bias term is considered as the last added feature

# Note: Here the bias term is being considered as the last added feature
class Perceptron(Classifier):
    ''' Implements the Perceptron inherited from Classifier For Classification... '''

    def __init__(self, lembda=0):
        """
            lembda= Regularization parameter...
        """

        Classifier.__init__(self, lembda)

        pass

    def hypothesis(self, X, theta):
        '''
            Computes the hypothesis for over given input examples (X) and parameters (thetas).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix
                theta: Must be a d-dimensional vector
            Return:
                The computed hypothesis
        '''
        
        # Your Code here
        #print 'X is :',X
        #print 'theta is :',theta
        return np.dot(X,theta)

    def cost_function(self, X, Y, theta):
        '''
            Computes the Cost function for given input data (X) and labels (Y).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix of inputs
                theata: must  d X 1-dimensional vector for representing vectors
                Y: Must be n X 1-dimensional label vector
                
            Return:
                Returns the cost of hypothesis with input parameters 
        '''

        
        # Your Code here
        #prod=Y*self.hypothesis(X,theta)
        #print 'shape of Y is',Y
        #print 'shape of hyp',self.hypothesis(X,theta)
        prod=Y*self.hypothesis(X,theta)
        #print 'prod is:',prod
        prod=-prod
        boolean=prod<0
        prod[boolean]=0
        mean=np.mean(prod)
        mean=mean/2
        theta_sqr=[]
        for i in range(0,len(theta)-1):
            temp=theta[i]**2
            theta_sqr.append(temp)
        theta_sqr=np.array(theta_sqr)
        cost=mean+((self.lembda/2)*(np.sum(theta_sqr)))
        return cost


    def derivative_cost_function(self, X, Y, theta):
        '''
            Computes the derivates of Cost function w.r.t input parameters (thetas)  
            for given input and labels.

            Input:
            ------
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix of inputs
                theata: must  d X 1-dimensional vector for representing vectors
                Y: Must be n X 1-dimensional label vector
            Returns:
            ------
                partial_thetas: a d X 1-dimensional vector of partial derivatives of cost function w.r.t parameters..
        '''
        
        # Your Code here
        X=np.array(X)
        prod=Y*self.hypothesis(X,theta)
        Y=Y.T
        rows,columns=X.shape
        derivatives=np.zeros(columns)
        for i in range(0,columns):
            col_vec=X[:,i].copy()
            col_vec=col_vec.reshape(1,rows)
            col_vec=-1*Y*col_vec
            indexes=np.logical_not(prod.T<0)
            col_vec[indexes]=0
            term1=np.sum(col_vec)
            term2=(1/(2.0*rows))
            term3=self.lembda*theta[i]
            derivatives[i]=term1*term2+term3
    
        return derivatives

    def train(self, X, Y, optimizer):
        ''' Train classifier using the given 
            X [m x d] data matrix and Y labels matrix
            
            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            optimizer: an object of Optimizer class, used to find
                       find best set of parameters by calling its
                       gradient descent method...
            Returns:
            -----------
            Nothing
            '''

        nexamples, nfeatures = X.shape
        # Your Code here
        self.theta=optimizer.gradient_descent(X,Y,self.cost_function,self.derivative_cost_function)

    def predict(self, X):
        """
        Test the trained perceptron classifier result on the given examples X
        
                   
            Input:
            ------
            X: [m x d] a matrix of m  d-dimensional test examples.
           
            Returns:
            -----------
            pclass: the predicted class for the given set of examples, i.e. to which it belongs
        """

        # Your Code here
        num_test = X.shape[0]
        
        
        if len(self.theta)-X.shape[1] ==1:
            # append 1 at the end of each example for the bias term
            X=np.hstack((X,np.ones((X.shape[0],1))))
        
        
        res=self.hypothesis(X,self.theta)
        boolean1=res>0.5
        boolean2=res<0.5
        res[boolean1]=1
        res[boolean2]=-1
        return res
