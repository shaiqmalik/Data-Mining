#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#

# A Logistic Regression algorithm with regularized weights...

from classifier import *

#Note: Here the bias term is considered as the last added feature 

class LogisticRegression(Classifier):
    ''' Implements the LogisticRegression For Classification... '''
    def __init__(self, lembda=0.001):        
        """
            lembda= Regularization parameter...            
        """
        Classifier.__init__(self,lembda)                
        
        pass
    def sigmoid(self,z):
        """
            Compute the sigmoid function 
            Input:
                z can be a scalar or a matrix
            Returns:
                sigmoid of the input variable z
        """

        # Your Code here
        boolean1=z>0.5
        boolean2=z<0.5
        z[boolean1]=1
        z[boolean2]=0
        return z        
    
    
    def hypothesis(self, X,theta):
        '''
            Computes the hypothesis for over given input examples (X) and parameters (thetas).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix
                theta: Must be a d-dimensional vector
            Return:
                The computed hypothesis
        '''
        
        # Your Code here
        #print 'X dtype',X.dtype
        #print 'theta dtype',theta.dtype
        #print 'X shape',X.shape
        #print 'theta shape',theta.shape
        #X=X.astype(float)
        theta=theta.astype(float)
        return 1/(1+np.exp(-np.dot(X,theta)))

        
    def cost_function(self, X,Y, theta):
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
        prod1=-Y*np.log(self.hypothesis(X,theta))
        prod2=(1-Y)*np.log(1-self.hypothesis(X,theta))
        diff=prod1-prod2
        mean=np.mean(diff)
        theta_sqr=[]
        for i in range(0,len(theta)-1):
            temp=theta[i]**2
            theta_sqr.append(temp)
        theta_sqr=np.array(theta_sqr)
        cost=mean+((self.lembda/2)*(np.sum(theta_sqr)))
        return cost

    def derivative_cost_function(self,X,Y,theta):
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
        nexamples=float(X.shape[0])
        temp=self.hypothesis(X,theta)
        res=np.dot((temp-Y).T,X)
        dtheta=(res/nexamples)
        s=theta.copy()
        s[-1]=0
        dtheta=dtheta+(self.lembda*s.T)
        return dtheta.T

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
        
        # Your Code here 
        # Use optimizer here

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
        
        num_test = X.shape[0]
        if len(self.theta)-X.shape[1] ==1:
            # append 1 at the end of each example for the bias term
            X=np.hstack((X,np.ones((X.shape[0],1))))
        
        # Your Code here
        res=self.hypothesis(X,self.theta)
        res=self.sigmoid(res)
        return res