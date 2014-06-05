''' external imports '''
import sys, os, matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy import interp
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from collections import defaultdict

''' internal imports '''
from metaSVM import metaSVM
import params as p

# kernel function for SVM
kernel_func = 'linear' # rbf

###############################################################################
def train_metaSVM(train_dict, regime_weights, regime_c_terms):
    ''' return a metaSVM object trained on the given data mixture '''
    
    # create mixed SVM
    meta_svm = metaSVM(regime_weights)
    
    # train on each regime
    for regime in [i for i in range( len(train_dict)/2 ) ]:
        # training data
        train_data, train_labels = normalize(train_dict[regime, "data"]), train_dict[regime, "states"]
        
        # make classifier
        curr_clf  = SVC( kernel='linear', probability=True, C=regime_c_terms[regime], cache_size=500 )
        #curr_clf = LogisticRegression( penalty='l1', fit_intercept=True, C=regime_c_terms[regime] )
        
        # train & add to meta SVM
        curr_clf.fit(train_data, train_labels)
        meta_svm.add_trained_SVM(curr_clf, regime)
    
    return meta_svm

###############################################################################
def metaSVM_pow(test_data, test_states, svm_mix, s, t, svm_dict=None):
    ''' test and evaluate power of the given mix-svm on the test data & states '''
    
    # estimate power of svm_mix on given test data
    probs, svm2pos_probs = svm_mix.predict_proba(normalize(test_data)) # class-probs & the pos-class probs per SVM
    
    # generate ROC curve
    fpr, tpr, thresholds = roc_curve(test_states, probs[:,1])
    plot_fpr = np.linspace(0, 1, 100)
    plot_tpr = interp(plot_fpr, fpr, tpr)
    plot_tpr[0],plot_tpr[-1] = 0.0, 1.0
     
    # find the threshold for 5% FPR
    five_pc_fpr_thresh = 0
    for i, rate in enumerate(fpr):
        if(rate >= 0.05):
            five_pc_fpr_thresh = thresholds[i]
            break
    
    # scatter pos-class probabilities
    classifying_svm = plot_pos_class_probs(five_pc_fpr_thresh, plot_tpr[5], test_states, svm2pos_probs, s, t)
    
    # guess s given t, if specific SVMs available
    frac_s_str = None
    if(svm_dict != None):
        tot_tp, tot_tp_correct = 0,0
        for i in range(len(test_states)):
            if(test_states[i] == 1 and probs[i,1] >= five_pc_fpr_thresh):
                # true positive
                tot_tp += 1
                max_p, max_p_s = -0.1,0.0
                for s_try in p.selection:
                    point = test_data[i] / np.linalg.norm( test_data[i] )
                    try_p = svm_dict[s_try,t].predict_proba( np.array([point]) )
                    if(try_p[0,1] >= max_p): max_p, max_p_s = try_p[0,1], s_try
                
                if(max_p_s == s): tot_tp_correct += 1
    
        frac_tp_correct = float(tot_tp_correct) / float(tot_tp) 
        frac_s_str = "%g=(%i/%i)" % (frac_tp_correct, tot_tp_correct, tot_tp)
    
    # find frac of points classified (above 5% FPR) by each SVM
    c_svm_counts = defaultdict(int)
    for c_svm in classifying_svm: c_svm_counts[c_svm] += 1
    svm2frac = {}
    for regime in svm_mix.trained_SVMs.keys():
        svm2frac[regime] = float(c_svm_counts[regime]) / (len(classifying_svm) - c_svm_counts[None]) # TP-regime / all TP
    
    # plot ROC
    plot_roc(plot_fpr, plot_tpr, svm_mix.trained_SVMs[0].coef_[0,:], 0, s, t)
    
    # return power & frac-classifying-SVM
    return plot_tpr[5], svm2frac, frac_s_str

###############################################################################
def plot_pos_class_probs(p_thresh, power, test_states, svm2pos_probs, s, t):
    """ find fraction of points (above 5% FPR) classified by 0/1 SVM, 
        and plot positive class probabilities as scatter
        
        Returns: list [classifying_svm] of size [test_states], where "None" represents points classified as CONTROL
        and "0","1","2".. represent points classified as CASE, where the entry is the index of the classifying SVM
    """
    classifying_svm = [] # returned list
    x_case, y_case, x_cont, y_cont = [],[],[],[] # x is SVM-0 (near-fixation), y is SVM-1 (post-fixation)
    
    # get points to plot
    for i in range(len(test_states)):
        # find the SVM with max prob for current point
        max_prob, max_prob_svm = -1, -1
        for svm_ind in svm2pos_probs.keys():
            if(svm2pos_probs[svm_ind][i] > max_prob):
                max_prob = svm2pos_probs[svm_ind][i]
                max_prob_svm = svm_ind

        # remember classifying SVM
        if(max_prob >= p_thresh and test_states[i] == 1):
            classifying_svm.append(max_prob_svm) # true positive point 
        else:
            classifying_svm.append(None) # true/false negative, or false positive point 
        
        # get case/cont separated points for scatter
        if(test_states[i] == 1):
            x_case.append(svm2pos_probs[0][i])
            y_case.append(svm2pos_probs[1][i])
        elif(test_states[i] == -1):
            x_cont.append(svm2pos_probs[0][i])
            y_cont.append(svm2pos_probs[1][i])
    
    # plot scatter
    matplotlib.rc('text', usetex=True)
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.12)
    ax = fig.add_subplot(111)
    
    # color over the 5% FPR threshold
    ax.fill_between([0., 0., 1.01, 1.01, p_thresh, p_thresh, 0.],[p_thresh, 1.01, 1.01, 0., 0., p_thresh, p_thresh], color='0.72')
    
    # scatter points representing samples
    ax.scatter(x_case, y_case, marker='+', facecolor='red', edgecolor='red', label=r"$\mathbf{Selection}$") # under selection (case)
    ax.scatter(x_cont, y_cont, marker='o', facecolor='none' , edgecolor='black', label=r"$\mathbf{Neutral}$") # under neutrality (control)
    
    # add legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,loc='lower center', prop={'size':11}, ncol=2, columnspacing=1.2, labelspacing=0.2, handletextpad=0.5, handlelength=2.25) #, fancybox=True, shadow=True)
    
    # add y=x line
    ax.plot([0, 1], [0, 1], '--', color='blue', label=r"$\mathbf{SVM\mbox{-}border}$")
    
    # some info
    fig.text(0.15, 0.025, r'$\mathbf{Power=%g}$' % power, fontsize=12)
    fig.text(0.70, 0.025, r'$\mathbf{p-thresh=%.3f}$' % p_thresh, fontsize=12)

    # plot logistics
    ax.set_xlabel(r"$\mathbf{Near\mbox{-}Fixation\, SVM}$")
    ax.set_ylabel(r"$\mathbf{Post\mbox{-}Fixation\, SVM}$")
    ax.set_title(r"$\mathbf{Non\mbox{-}Neutral\, (class)\, Probabilities\, (s=%g,\tau=%i)}$" % (s,t))
    ax.set_xlim(0,1.01)
    ax.set_ylim(0,1.01)
    
    # directory and filename
    pos_probs_scatter_dir = "pos_probs"
    if(not os.path.isdir(pos_probs_scatter_dir)):
        os.mkdir(pos_probs_scatter_dir)
    save_to = pos_probs_scatter_dir + "/s%g.t%i.pos.probs.png" % (s,t)
    
    # save file
    plt.savefig(save_to, dpi=400)
    plt.close(fig)
    
    return classifying_svm
 
###############################################################################
def svc_search(datas, states, K, s, t):
    ''' For given (s,t) trains all possible SVMs using (t_train,s_train) training data, 
        and estimates power of those on the parameters (dataset) given. 
    '''
    
    print "Searching SVM for (s=%g, t=%i): " % (s,t) 
    train_st_2_pow = {} # power using each training dataset (s,t)
    train_st_pow = [] # same as list 
    
    # prep
    best_pow_t, best_pow_s, best_pow, own_pow = 0, 0.0, 0.0, 0.0
    test_data, test_states = datas[s,t], states[s,t]
    
    for s_train in p.selection:
        for t_train in p.times:
            # for each (s,t), calc ROC of SVM trained on data from 
            # (s_train,t_train) but tested on data from (s_test, t_test)
            
            train_data = datas[s_train,t_train]
            train_states = states[s_train,t_train]
            
            if(s_train == s and t_train == t):
                mean_fpr, mean_tpr, svm_clf, w = svm_cross_val_ROC(test_data, test_states, K, s, t, False, 1.0) # own (t,s), use c.v. 
                p, own_p = mean_tpr[5], mean_tpr[5] 
            else:
                p, w = svm_ROC(train_data, train_states, test_data, test_states, K, t, s, False) # other (t,s), no c.v. 
            
            # remember
            train_st_2_pow[s,t] = p
            train_st_pow.append(p)
            
            if(p > best_pow):
                best_pow = p
                best_pow_t = t_train
                best_pow_s = s_train
        
    print "\t" + "Own  (s=%g, t=%i) power (5%% FPR) SVM: %g" % (s,t, own_pow)
    print "\t" + "Best (s=%g, t=%i) power (5%% FPR) SVM: %g\n" % (best_pow_s, best_pow_t, best_pow)
    
    return train_st_2_pow, train_st_pow

###############################################################################
def svm_ROC(A_train, y_train, A_test, y_test, K,  t, s, plot_ROC):
    ''' SVC where SVM is trained on (A_train, y_train) and tested on (A_test, y_test). 
        ROC is calculated & plotted.
        Returns:
            1) mean power (at 0.05 FPR).
            2) array with classification-status of each data point in A, s.t. non-zero entries are well-classified.
    '''
    
    # normalize (training & test) data
    A_train_norm = normalize(A_train)
    A_test_norm  = normalize(A_test)
    
    # prep SVM classifier
    svm_clf = SVC( kernel='linear', probability=True, cache_size=500 ) # default C=1.0
    
    # train SVM
    svm_clf.fit(A_train_norm, y_train)
    
    # weight vector for clustering
    real_w = svm_clf.coef_[0,:]
    
    # classify test-data & find well-classified points
    class_predictions = svm_clf.predict(A_test_norm) # class-predictions
    well_classified = np.zeros( len(A_test) )
    for i in range(0, len(A_test_norm)):     
        if(y_test[i] == class_predictions[i]):  
            well_classified[i] = 1
            
    # classify test-data & generate ROC curve
    probs = svm_clf.predict_proba(A_test_norm) # class-probs
    fpr, tpr, thresholds = roc_curve(y_test, probs[:,1])
    plot_fpr = np.linspace(0, 1, 100)
    plot_tpr = interp(plot_fpr, fpr, tpr)
    plot_tpr[0],plot_tpr[-1] = 0.0, 1.0
    
    # plot ROC curve
    if(plot_ROC): plot_roc(plot_fpr, plot_tpr, real_w, K, s, t)

    return plot_tpr[5], well_classified, real_w

###############################################################################
def svm_cross_val_ROC(A_norm, y, K,  s, t, plot_ROC, c):
    ''' 
    SVC on data matrix A and state vector y. The mean ROC (using K-fold cross validation) is calculated & plotted.
    Returns:
        1) mean power (at 0.05 FPR).
        2) array with classification-status of each data point in A, s.t. non-zero entries are well-classified.
    '''      
    
    ############ classifier ############
    clf = SVC( kernel=kernel_func, probability=True, C=c, cache_size=500 ) # clf = LogisticRegression(penalty='l1', fit_intercept=True, C=c)
    
    ############### prep ###############
    well_classified = np.zeros( len(A_norm) )
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    cv = StratifiedKFold(y, indices=False, n_folds=K) # c.v. partition

    ####### mean (c.v) ROC curve #######
    for i, (train, test) in enumerate(cv):

        # SVM training
        clf.fit(A_norm[train],y[train])
        
        # SVM classification, class probabilities
        probs_lin = clf.predict_proba(A_norm[test]) 
        
        # ROC for current c.v. partition 
        fpr, tpr, thresholds = roc_curve(y[test], probs_lin[:,1])  
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        
        if(False):
            # find well-classified points
            class_predictions = clf.predict(A_norm[test])
            cp_ind = 0 # index in class_predictions array
            for i in range(0, len(A_norm)): # index in data matrix A 
                if(test[i] == True):
                    if(y[i] == class_predictions[cp_ind]):
                        well_classified[i] = 1 # well-predicted point
                    cp_ind += 1 # increment 
      
    # finish ROC 
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0

    # SVM trained on complete data
    clf.fit(A_norm, y)

    if(False):
        # only for SVMs 
        class_predictions = clf.predict(A_norm) # class-predictions
        SVs, SV_ind = clf.support_vectors_, clf.support_
        w, inter = clf.coef_[0,:], clf.intercept_[0]
        well_class = 0
        for i in range(len(A_norm)):
            if(y[i] == class_predictions[i]):
                well_class += 1
                if(y[i] * float(np.dot(w, A_norm[i]) + inter) > 0):
                    print "sign diff"
                
                if(i in SV_ind):
                    print np.dot(w, A_norm[i]) + inter
                    raw_input("Press Enter to continue...")
        
        print "num of well-class %i" % well_class 
        
        print "number of SVs:"
        print clf.n_support_
        raw_input("Press Enter to continue...")
        for i in range(len(SVs)):
            print SVs[i,:]
            print np.dot(w, SVs[i,:]) + inter
            raw_input("Press Enter to continue...")
        
        sys.exit()
        ##### done
    
    # feature weights 
    real_w = None
    if(kernel_func == 'linear'):
        real_w = clf.coef_[0,:]
    
    # plot mean ROC curve
    if(plot_ROC): plot_roc(mean_fpr, mean_tpr, real_w, K, s, t) 
    
    return mean_fpr, mean_tpr, clf, real_w
    
###############################################################################
def plot_roc(mean_fpr, mean_tpr, f_weights, K, s, t): 
    
    # directory and filename
    ROC_dir = "ROCs"
    if(not os.path.isdir(ROC_dir)):
        os.mkdir(ROC_dir)
    save_to = ROC_dir + "/s%g.t%i.scaled.sfs.svm.png" % (s,t)
    
    # prep plot
    matplotlib.rc('text', usetex=True)
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.35)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    # plot ROC curve
    ax1.plot(mean_fpr, mean_tpr, 'b-', label=r"$\mathbf{Linear\, kernel\, (auc = %0.2f)}$" % auc(mean_fpr, mean_tpr), lw=1.2)
    ax1.plot([0, 1], [0, 1], '--', color='black', label=r"$\mathbf{Luck}$")
    ax1.set_xlim([0.0,1.0])
    ax1.set_ylim([0.0,1.01])
    ax1.axvline(x=0.05, linewidth=1.3, color='grey', linestyle='dashed')
    ax1.set_xlabel(r"$\mathbf{FPR}$")
    ax1.set_ylabel(r"$\mathbf{TPR}$")
    
    ax1.set_title(r"$\mathbf{Mean\, (%i\mbox{-}fold\, c.v.)\, ROC\, of\, SVC\, on\, scaled\mbox{-}SFS\, (s = %s,\tau = %i)}$" % (K,s,t))
    ax1.legend(loc="lower right", prop={'size':10})
    
    # plot feature weights  
    x = [ float(i) / float( len(f_weights) ) for i in range(len(f_weights)) ]
    b_width = 1.0 / float( len(x) )
    ax2.bar(x ,f_weights, b_width, linewidth=0)
    ax2.set_title(r"$\mathbf{SVM \, (linear-kernel) \, Feature \, Weights}$")
    ax2.set_xlim(-0.01,1.01)
    
    plt.savefig(save_to, dpi=400)
    plt.close(fig)

###############################################################################
def normalize(A):
    """ normalize data-points in matrix A such that 2norm^2 == 1 """
    A_norm = np.array(A)
    for i in range( len(A) ):
        A_norm[i] = A[i] / np.linalg.norm( A[i] ) # or ( A[i] - np.mean(A[i]) ) / np.linalg.norm( A[i] - np.mean(A[i]) )
    
    return A_norm
