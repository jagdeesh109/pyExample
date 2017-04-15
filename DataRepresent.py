'''
Created on Apr 15, 2017

@author: Jagdeesh.Gughalot
'''
import sys
from sklearn.manifold import Isomap
from sklearn.datasets import load_digits 
from sklearn.decomposition import  PCA
import matplotlib.pyplot as plt
import matplotlib.figure as fig
class DataRepresentation :
    
    def toyData(self):
       
#        Data representation
#        Everything is a numpy array (or a scipy sparse matrix)! 
       
        digits = load_digits();
        
#         print the shape of the images.
        
        
        
        print("images shape: %s" % str(digits.images.shape))
        print("targets shape: %s" % str(digits.target.shape))
        
#     plot the array    
        
        plt.matshow(digits.images[0], cmap= 'gray');
      #  plt.show();
        
        digits.target
         

       # prepare the data
        
        X = digits.data.reshape(-1,64)
        
        print("data of x")
        print(X.shape )
        
        
        y = digits.target
        print(y.shape)
        
        
# We have 1797 data points, each an 8x8 image -> 64 dimensional vector.

#X.shape is always (n_samples, n_feature)
        
        
        print(X)
        
        # Principal Component Analysis (PCA)
        
        # nstantiate the model. Set parameters.
        pca = PCA(n_components=2)
        
        #Fit the model.
        pca.fit(X)
        
        # Apply the model. For embeddings / decompositions, this is transform.

        X_pca = pca.transform(X,None);
        X_pca.shape
        
        plt.figure(figsize=(16, 10));
        
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y);   
        
        plt.show()
        
        # print the pca mean and component
        
        print(pca.mean_.shape)
        print(pca.components_.shape)
        
        fix,ax = plt.subplots(1,3)
        ax[0].matshow(pca.mean_.reshape(8, 8), cmap='gray')
        ax[1].matshow(pca.components_[0, :].reshape(8, 8), cmap='gray')
        ax[2].matshow(pca.components_[1, :].reshape(8, 8), cmap='gray')
        
        #Isomap
        
        # Instantiate the model. Set parameters.
       
        isomap = Isomap(n_components=2, n_neighbors=20)
        
       # Fit the model.
       
      # ref :  http://glowingpython.blogspot.in/2012/05/manifold-learning-on-handwritten-digits.html
      # http://cs231n.github.io/python-numpy-tutorial/
      # https://github.com/amueller
      # https://www.learnpython.org/en/Classes_and_Objects
      # http://amueller.github.io/sklearn_tutorial/#/13  
        
        
            
        
        
        
        
        
        
        
        
        
        
        