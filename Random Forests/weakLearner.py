#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#


#---------------Instructions------------------#

# You will be writing a super class named WeakLearner
# and then will be implmenting its sub classes
# RandomWeakLearner and LinearWeakLearner. Remember
# all the overridded functions in Python are by default
# virtual functions and every child classes inherits all the
# properties and attributes of parent class.

# Your task is to  override the train and evaluate functions
# of superclass WeakLearner in each of its base classes. 
# For this purpose you might have to write the auxiliary functions as well.

#--------------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#


import numpy as np
import scipy.stats as stats

class WeakLearner: # A simple weaklearner you used in Decision Trees...
    """ A Super class to implement different forms of weak learners...

    """
    best_split=0
    best_split_cols=[]
    def __init__(self):
        """
        Input:
            

        """
        #print "   "        
        pass

    def train(self,feat, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            feat: a contiuous feature
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        Xlidxf=[]
        Xridxf=[]
        min_score=+np.inf
        best_split=-np.inf
        best_split_col=-1
        for i in range(0,nfeatures):
            print 'calling evaluate_numerical_attribute'
            split,ent,lidx,ridx=self.evaluate_numerical_attribute(X[:,i],Y)
            if(score<min_score):
                min_score=ent
                best_split=split
                best_split_col=i
                Xlidxf=lidx
                Xridxf=ridx
        self.best_split=best_split
        self.best_split_cols=best_split_col
        #---------End of Your Code-------------------------#
        return min_score, Xlidxf,Xridxf
    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        if X.shape[0] == 1:
            X=X.flatten()
        if(X[self.best_split_cols]<self.best_split):
            return True
        else:
            return False
        #---------End of Your Code-------------------------#
    def evaluate_numerical_attribute(self,feat, Y):
        '''
            Evaluates the numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            feat: a contiuous feature
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        
        classes=np.unique(Y)
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        # Same code as you written in DT assignment...
        self.classes=classes
        nclasses=len(self.classes)
        sidx=np.argsort(feat)
        f=feat[sidx] # sorted features
        sY=Y[sidx] # sorted features class labels...
        mid_points_temp=np.zeros(len(f)-1)
        class_counts=np.zeros(len(classes))
        
        for c in range(0,len(classes)):
            class_counts[c]=np.sum(Y==classes[c])
        for i in range(0,len(f)-1):
            mid_points_temp[i]=(f[i]+f[i+1])/2
        mid_points=np.unique(mid_points_temp)
        split_counts=np.zeros((len(mid_points),len(classes)))
        for p in range(0,len(mid_points)):
            for j in range(0,len(f)):
                class_num=0
                for k in range(0,len(classes)):
                    if(sY[j]==classes[k]):
                        class_num=k
                        break
                if(f[j]<mid_points[p]):
                    split_counts[p][class_num]=split_counts[p][class_num]+1
        rows,col=split_counts.shape
        split_entropies=np.zeros(len(mid_points))
        gains=np.zeros(len(mid_points))
        Xlidx=[]
        Xridx=[]
        for l in range(0,rows):
            denom=np.sum(split_counts[l,:])
            denom2=np.sum(class_counts)-denom
            log_sum=0
            log_sum2=0
            for m in range(0,col):
                if(split_counts[l][m]!=0):
                    log_sum=log_sum+((split_counts[l][m]/denom)*(np.log2(split_counts[l][m]/denom)))
                if((class_counts[m]-split_counts[l][m])!=0):
                    log_sum2=log_sum2+(((class_counts[m]-split_counts[l][m])/denom2)*(np.log2((class_counts[m]-split_counts[l][m])/denom2)))
                    
            log_sum=-1*(log_sum)
            log_sum2=-1*(log_sum2)
            split_entropy=((denom/float(denom+denom2))*log_sum)+((denom2/float(denom+denom2))*log_sum2)
            split_entropies[l]=split_entropy
        split_entropies=split_entropies[split_entropies>0]
        min_split_entropy=np.argmin(split_entropies)
        split=mid_points[min_split_entropy]
        mingain=split_entropies[min_split_entropy]
        for s in range(0,len(f)):
            if(f[s]<split):
                Xlidx.append(s)
            else:
                Xridx.append(s)
        Xlidx=np.array(Xlidx)
        Xridx=np.array(Xridx)  
        
        #---------End of Your Code-------------------------#
            
        return split,mingain,Xlidx,Xridx

class RandomWeakLearner(WeakLearner):  # Axis Aligned weak learner....
    """ An Inherited class to implement Axis-Aligned weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=+np.inf, nrandfeat=None):
        """
        Input:
            nsplits = How many nsplits to use for each random feature, (if +inf, check all possible splits)
            nrandfeat = number of random features to test for each node (if None, nrandfeat= sqrt(nfeatures) )
        """
        WeakLearner.__init__(self) # calling base class constructor...        
        self.nsplits=nsplits
        self.nrandfeat=nrandfeat
        pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            X: a [m x d]  features matrix
            Y: a [m x 1] labels matrix
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        
        if(not self.nrandfeat):
            self.nrandfeat=np.round(np.sqrt(nfeatures))
        scores=[]
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        minscore=+np.inf
        best_split=-np.inf
        best_split_col=-1
        bXl=[]
        bXr=[]
        idx=np.random.randint(0,nfeatures,self.nrandfeat)
        i=0
        while i<self.nrandfeat:
            if(self.nsplits!=+np.inf):
                split,min_ent,Xlidx,Xridx=self.findBestRandomSplit(X[:,idx[i]],Y)
            else:
                split,min_ent,Xlidx,Xridx=self.evaluate_numerical_attribute(X[:,idx[i]],Y)
            if(min_ent<minscore):
                minscore=min_ent
                bXl=Xlidx
                bXr=Xridx
                best_split=split
                best_split_col=idx[i]
            i=i+1
        self.best_split=best_split
        self.best_split_cols=best_split_col
        
        #---------End of Your Code-------------------------#
        return minscore, bXl,bXr

    def findBestRandomSplit(self,feat,Y):
        """
            
            Find the best random split by randomly sampling "nsplits"
            splits from the feature range...

            Input:
            ----------
            feat: [n X 1] nexamples with a single feature
            Y: [n X 1] label vector...

        """
        frange=np.max(feat)-np.min(feat)

        #import pdb;         pdb.set_trace()
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        #print 'calling calculate best split function'
        random_splits=np.random.rand(1,self.nsplits)
        random_splits=random_splits*frange
        random_splits=random_splits+np.min(feat)
        min_ent=+np.inf
        best_split=-np.inf
        #print 'shape of random splits:',random_splits.shape
        #print ' random split is',random_splits.flatten()
        random_splits=random_splits.flatten()
        for split in random_splits:
            #print 'calculating entropy for:',split
            #print 'split is',split
            mship=feat<split
            temp=self.calculateEntropy(Y,mship)
            if(temp<min_ent):
                min_ent=temp
                best_split=split
        Xlidx=[]
        Xridx=[]
        for s in range(0,len(feat)):
            if(feat[s]<best_split):
                Xlidx.append(s)
            else:
                Xridx.append(s)
            
        #---------End of Your Code-------------------------#
        return best_split, min_ent, Xlidx, Xridx
    def calculateEntropy(self,Y, mship):
        """
            calculates the split entropy using Y and mship (logical array) telling which 
            child the examples are being split into...

            Input:
            ---------
                Y: a label array
                mship: (logical array) telling which child the examples are being split into, whether
                        each example is assigned to left split or the right one..
            Returns:
            ---------
                entropy: split entropy of the split
        """

        lexam=Y[mship]
        rexam=Y[np.logical_not(mship)]

        pleft= len(lexam) / float(len(Y))
        pright= 1-pleft

        pl= stats.itemfreq(lexam)[:,1] / float(len(lexam)) + np.spacing(1)
        pr= stats.itemfreq(rexam)[:,1] / float(len(rexam)) + np.spacing(1)

        hl= -np.sum(pl*np.log2(pl)) 
        hr= -np.sum(pr*np.log2(pr)) 

        sentropy = pleft * hl + pright * hr

        return sentropy



# build a classifier ax+by+c=0
class LinearWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D line based weak learner using 
        a random set of features from the given set of features...


    """
    a=0
    b=0
    c=0
    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...
            
        """
        RandomWeakLearner.__init__(self,nsplits)
        
        pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 
            
            Input:
            ---------
            X: a [m x d] data matrix ...
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        minscore=+np.inf
        best_split=-np.inf
        best_split_col=-1
        for i in range(0,self.nsplits):
            features=np.random.randint(0,nfeatures,2)
            a=np.random.normal()
            b=np.random.normal()
            c=np.random.normal()
            ans=(X[:,features[0]]*a)+(X[:,features[1]]*b)+c
            mship=(ans<=0)
            ent=self.calculateEntropy(Y,mship)
            if(ent<minscore):
                best_split_col=features
                self.a=a
                self.b=b
                self.c=c
                minscore=ent
                best_split=ans
        
        bXl=np.argwhere(best_split<=0)
        bXr=np.argwhere(best_split>0)
        bXl=bXl.flatten()
        bXr=bXr.flatten()
        #print 'bXl:',bXl
        #print 'bXr:',bXr
        #print "best split cols are:",best_split_col
        self.best_split_cols=best_split_col
        #print "best split cols are:",self.best_split_cols[0]
        #print "best split cols are:",self.best_split_cols[1]
        #---------End of Your Code-------------------------#

        return minscore, bXl, bXr
    
    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        f1=self.best_split_cols[0]
        f2=self.best_split_cols[1]
        #print 'the shape of X is:',X.shape
        if X.ndim==1:
            ans=(X[:,f1]*self.a)+(1*self.b)+self.c
        else:
            ans=(X[:,f1]*self.a)+(X[:,f2]*self.b)+self.c
        if(ans<0):
            return True
        else:
            return False
            
        
        #---------End of Your Code-------------------------#
        
class ConicWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D line based weak learner using 
        a random set of features from the given set of features...


    """
    a=0
    b=0
    c=0
    d=0
    e=0
    f=0
    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...
            
        """
        RandomWeakLearner.__init__(self,nsplits)
        
        pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 
            
            Input:
            ---------
            X: a [m x d] data matrix ...
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        minscore=+np.inf
        best_split=-np.inf
        best_split_col=-1
        best_bounds=[]
        for i in range(0,self.nsplits):
            features=np.random.randint(0,nfeatures,2)
            a=np.random.normal()
            b=np.random.normal()
            c=np.random.normal()
            d=np.random.normal()
            e=np.random.normal()
            f=np.random.normal()
            ans=((X[:,features[0]]**2)*a)+((X[:,features[1]]**2)*b)+((X[:,features[0]]*X[:,features[1]])*c)+(X[:,features[0]]*d)+(X[:,features[1]]*e)+f
            bounds=np.random.normal(size=(2,1))
            if(np.random.random(1)<0.5):
                bounds[0]=-np.inf
            mship= np.logical_and(ans>=bounds[0],ans<bounds[1])
            ent=self.calculateEntropy(Y,mship)
            if(ent<minscore):
                best_split_col=features
                self.a=a
                self.b=b
                self.c=c
                self.d=d
                self.e=e
                self.f=f
                minscore=ent
                best_split=ans
                best_bounds=bounds
        bXl=np.argwhere(best_split<=0)
        bXr=np.argwhere(best_split>0)
        bXl=bXl.flatten()
        bXr=bXr.flatten()
        self.best_split_cols=best_split_col
        self.best_split=best_bounds
        self.score=minscore
        
        
        #feature1=np.random.randint(nfeatures)
        #feature2=np.random.randint(nfeatures)
        #minscore,bXl,bXr=self.findBestRandomCone(X[:,feature1],X[:,feature2],Y)
        #self.best_split_cols.append(feature1)
        #self.best_split_cols.append(feature2)
        #---------End of Your Code-------------------------#

        return minscore, bXl, bXr

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        f1=self.best_split_cols[0]
        f2=self.best_split_cols[1]
        X=X.flatten()
        ans=((X[f1]**2)*self.a)+((X[f2]**2)*self.b)+((X[f1]*X[f2])*self.c)+(X[f1]*self.d)+(X[f2]*self.e)+self.f
        #print 'ans is:',ans
        #print 'logical and is:',np.logical_and(ans>=self.best_split[0],ans<self.best_split[1])
        if(ans>=self.best_split[0] and ans<self.best_split[1]):
            return True
        else:
            return False
        #    return True
        #else:
        #    return False
            
        
        #---------End of Your Code-------------------------#