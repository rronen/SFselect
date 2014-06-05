import numpy as np

class metaSVM:
    
    ############################### c'tor ################################
    def __init__(self, weights):
        self.trained_SVMs = {}
        self.svm_weights = weights
        
    ######################### add_trained_SVM ############################
    def add_trained_SVM(self, svm, key):
        ''' adds the given trained SVM to the mixSVM '''
        self.trained_SVMs[key] = svm
        
    ############################## predict ###############################
    def predict(self, test_data):
        ''' takes a list of vectors and returns an array of classification states.
            state is positive if >= 1 SVMs classified as positive, negative otherwise.
        '''
        preds = [-1] * len(test_data)
            
        # test data point on all trained SVMs
        for (key, svm) in self.trained_SVMs.iteritems():
            preds_curr = svm.predict(test_data) # current SVM class-predictions
            
            # update overall class predictions
            for i in range(len(test_data)):
                if(preds_curr[i] == 1):
                    preds[i] = 1
        
        return np.array(preds)
    
    ########################### predict_proba ############################
    def predict_proba(self, test_data):
        ''' takes a list of vectors and returns an array of class probabilities. 
            p(positive-class) = maximum positive-class probability over all internal SVMs.
            p(negative-class) = 1 - p(positive-class)
        '''
        svm2pos_probs = {}
        best_pos_probs = np.zeros(len(test_data)) # positive class prob 
        
        # test data on all trained SVMs
        for (key, svm) in self.trained_SVMs.iteritems():
            class_probs_curr = svm.predict_proba(test_data) * self.svm_weights[key] # get (weighted) SVM class-probabilities
            svm2pos_probs[key] = class_probs_curr[:,1]
            
            # update max positive class prob.
            for i in range(len(test_data)):
                if(class_probs_curr[i,1] > best_pos_probs[i]):
                    best_pos_probs[i] = class_probs_curr[i,1]
        
        negative_probs = 1.0 - best_pos_probs # negative class probabilities (conforming to the usual API)    
        rows = np.array([negative_probs, best_pos_probs])
        return rows.T, svm2pos_probs
