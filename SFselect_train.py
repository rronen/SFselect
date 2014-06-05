#!/usr/bin/env python

import sys, os, argparse
import numpy as np
import cPickle as pickle
from multiprocessing import Pool

''' internal imports '''
import learn, regimes, reader
import params as p

#  SFselect_train.py
#
#  This program applies supervised learning (SVM) to the site frequency spectrum.
#  It trains linear models of the SFS to best classify gemomic segments evolving 
#  under a 'hard sweep' model of positive selection from reigons evolving neutrally. 
#  Specifically,
#      1. reads site frequencies from sweep/neutral simulations.
#      2. generates binned & scaled SFS or XP-SFS vectors.
#      3. trains linear SVMs to best separate the two classes. 

####################################################################################
################################ Internal Params ###################################
####################################################################################

args = None

# data structures
st2pow      = {} # e.g. st2pow[s,t]
spec_svms   = {} # e.g. spec_svms[s,t]
datas       = {} # e.g. datas[s,t] 
states      = {} # e.g. states[s,t]
SFS_vectors = {} # e.g. SFS_vectors["cont2",0.02,1500,199]

# filenames for final trained models
gen_svm_file_to_write  = "general_svm.pck"
spec_svm_file_to_write = "specific_svms.pck"

####################################################################################
###################################### GO ##########################################
####################################################################################
def go(sfs_file, general_svm_file, specific_svm_file):
    global SFS_vectors, states, datas
    
    # read data
    SFS_vectors = reader.rescaled_binned_SFS(sfs_file)
    
    # place case/cont vectors for training and testing in 'datas' & 'states'  
    for s,t in [(s,t) for s in p.selection for t in p.times]:
        datas[s,t], states[s,t] = data_and_states_for(p.case_type, p.cont_type, [s], [t])


    # Apply ONE of the following for model training.

    # 1) specific_SVC
    # Trains (s,t)-specific models, and estimates their power at a fixed (e.g. 5%) 
    # FPR using cross validation.

    # 2) general_SVC_parition_data
    # Trains general models on a partitioned dataset. Typically 300 of 500 simulation 
    # instances (across all s,t) will be used in training, and the remaining 200 will 
    # be used in testing. There is there NO DANGER of overfitting, as the testing data 
    # is completely held out of training. However, only ~60% of the data is used for
    # training. This is practical, not VERY SLOW like the next training schema.
    
    # 3) general_SVC_exclude_test_set
    # Trains general models, where power at a specific (s,t) is estimated by training 
    # a model on all the data *excluding* data from that (s,t). There is there NO DANGER
    # of overfitting, as the testing data is completely held out of training. Also, the 
    # maximal amount of data for training is used. WARNING: VERY SLOW.

    # 4) pairwise_SVC
    # learns (s,t)-specific models, but computes their power across all (s,t) pairs.
    # The diagonal of the resulting power matrix (each model applied to its respective 
    # scenario) is what results from specific_SVC().

    # Note, In Ronen et al. 2013 (GENETICS) the results we report were obtained using (3). 
    # However, in reality (2) and (3) produce *very* similar results and thus we recommend 
    # using (2) for most practical cases. Also note that in all methods, the FULL DATASET is 
    # used to train the final models, which may be input to the program 'SFselect.py'.

    specific_SVC()
    #general_SVC_partition_data(general_svm_file, specific_svm_file) 
    #general_SVC_exclude_test_set(specific_svm_file)
    #pairwise_SVC()
        
####################################################################################
################################# Aux. Functions ###################################
####################################################################################

######################### General SVC - partitioned data ###########################
def general_SVC_partition_data(gen_svm_file=None, spec_svm_file=None):
    ''' Train general SVM according to regimes on partition 1, test on partition 2.
        Compute power at all (s,t)'s.
    '''
    global st2pow
    
    partition_at = 300 # 0 test all data, 300 train & test, 500 train all data
    pfile_text = []
    svm_dict = None
    
    if(gen_svm_file is None):
        ############ train meta-SVM on training-set #############
        sys.stdout.write('Training meta-SVM... ')
        
        # get training data
        train_mix = data_mix_for_SVM_general(p.first_sim, partition_at)
        
        # set equal regime-svm weights and error terms
        regimes_ind = [i for i in range(len(train_mix)/2)]
        svm_weights = dict( zip(regimes_ind, [1.0]*len(regimes_ind)) )
        svm_e_terms = dict( zip(regimes_ind, [0.1]*len(regimes_ind)) )
        
        # train
        meta_svm = learn.train_metaSVM(train_mix, svm_weights, svm_e_terms)        
        print 'done.\n'
        
        # if this was only for training, pickle and exit
        if(partition_at == p.last_sim):
            print 'Pickling SVM, then exiting...'
            with open(gen_svm_file_to_write, mode='wb') as gen_svm_fh:
                    pickle.dump(meta_svm, gen_svm_fh)
            sys.exit(0)
    else:
        ############# read SVM from pickled file ############## 
        sys.stdout.write("\nReading in cPickled SVM from: %s... " % gen_svm_file)
        with open("../" + gen_svm_file, mode='rb') as svm_fh:
            meta_svm = pickle.load(svm_fh)
        print "loaded.\n"
    
    if(spec_svm_file != None):
        # read (s,t)-specific SVM dict from pickled file 
        sys.stdout.write("\nReading in cPickled SVM-dict from: %s... " % spec_svm_file)
        with open("../" + spec_svm_file, mode='rb') as svm_fh:
            svm_dict = pickle.load(svm_fh)
        print "loaded.\n"
    
    #dotprod_and_decison_func_of(meta_svm, 0.02,700 , regimes.regime_of(0.02,700) )
    #dotprod_and_decison_func_of(meta_svm, 0.02,1400, regimes.regime_of(0.02,1400))
    
    ################# apply SVM to test data ##################
    for s,t in [(s,t) for s in p.selection for t in p.times]:
        
        # evaluate power on test set
        test_data, test_states = data_and_states_for(p.case_type, p.cont_type, [s], [t], partition_at, p.last_sim) # test data
        print "test-data: %i, test-states: %i" % (len(test_data), len(test_states))
        st2pow[s,t], svm2frac, frac_s_str = learn.metaSVM_pow(test_data, test_states, meta_svm, s, t, svm_dict)
        
        # report
        fracs = ""
        for i, frac in sorted(svm2frac.iteritems()): fracs += "%g\t" % frac
        if(frac_s_str == None):
            to_write = "%g\t%i\t%g\t%s" % (s, t, st2pow[s,t], fracs)
        else:
             to_write = "%g\t%i\t%g\t%s\t%s" % (s, t, st2pow[s,t], fracs, frac_s_str)
             
        print to_write
        pfile_text.append(to_write)
    
    # write power stats for file
    write_to_file('power_SVM_general.txt', pfile_text)
    
    ######## if none input, make & pickle the final meta-SVM ########
    if(gen_svm_file is None):
        # train meta-SVM on training-set
        sys.stdout.write('Training final meta-SVM... ')
        train_mix = data_mix_for_SVM_general(p.first_sim, p.last_sim)
        
        # set equal regime-svm weights and error terms
        regimes_ind = [i for i in range(len(train_mix)/2)]
        svm_weights = dict( zip(regimes_ind, [1.0]*len(regimes_ind)) )
        svm_e_terms = dict( zip(regimes_ind, [0.1]*len(regimes_ind)) )
        
        meta_svm = learn.train_metaSVM(train_mix, svm_weights, svm_e_terms)        
        print 'done.\n'
        
        with open(gen_svm_file_to_write, mode='wb') as gen_svm_fh:
                pickle.dump(meta_svm, gen_svm_fh)
    
######################### General SVC - excluded test set ##########################
def general_SVC_exclude_test_set(spec_svm_file=None):
    ''' Train the general SVM according to regimes on all data excluding the test data.
        Compute power at all (s,t)'s.
    '''
    global st2pow
    pfile_text = []
    
    ########## read (s,t)-specific SVMs from file ##########
    if(spec_svm_file != None):
        # read specific SVM dict from pickled file 
        sys.stdout.write("Reading in cPickled SVM-dict from: %s... " % spec_svm_file)
        with open("../" + spec_svm_file, mode='rb') as svm_fh:
            svm_dict = pickle.load(svm_fh)
        print "loaded.\n"
    
    ############## train & test on all (s,t) ###############
    for s,t in [(s,t) for s in p.selection for t in p.times]:        
        
        # training data, EXCLUDING test data
        print "training (excluded) on (%g,%i)" % (s,t)
        train_mix = data_mix_for_SVM_general_exclude(s,t) 
        
        svm_weights, svm_e_terms = {}, {}
        for i in range(len(train_mix)/2): svm_weights[i], svm_e_terms[i] = 1.0, 0.1
        
        meta_svm = learn.train_metaSVM(train_mix, svm_weights, svm_e_terms)
        
        # test
        test_data, test_states = datas[s,t], states[s,t] # test data
        st2pow[s,t], svm2frac, frac_s_str = learn.metaSVM_pow(test_data, test_states, meta_svm, s, t, svm_dict)
        
        # report
        fracs = ""
        for i, frac in sorted(svm2frac.iteritems()): fracs += "%g\t" % frac
        if(frac_s_str == None):
            to_write = "%g\t%i\t%g\t%s" % (s, t, st2pow[s,t], fracs)
        else:
             to_write = "%g\t%i\t%g\t%s\t%s" % (s, t, st2pow[s,t], fracs, frac_s_str) 
        print to_write
        pfile_text.append(to_write)
        
    write_to_file('power_SVM_general.txt', pfile_text)
    
    ########### make final meta-SVM and pickle ###########
    sys.stdout.write('Training final meta-SVM... ')
    train_mix = data_mix_for_SVM_general(p.first_sim, p.last_sim)
    
    # even svm-weights & error terms
    svm_weights, svm_e_terms = {}, {}
    for i in range(len(train_mix)/2):
        svm_weights[i], svm_e_terms[i] = 1.0, 0.1
    
    meta_svm = learn.train_metaSVM(train_mix, svm_weights, svm_e_terms)        
    print 'done.\n'
    
    with open(gen_svm_file_to_write, mode='wb') as gen_svm_fh:
            pickle.dump(meta_svm, gen_svm_fh)

################################### Specific SVC ###################################
def specific_SVC():
    ''' Compute power at different (s,t), of SVMs trained on that specific data.
        Power estimated by cross validation.
    '''
    
    global spec_svms, st2pow

    vects, labels  = [],[] # svm-vectors & full labels 
    
    process_pool = Pool(processes=min(len(p.c_grid), 5))
    
    for s,t in [(s,t) for s in p.selection for t in p.times]:
        
        # prep
        data, state = data_and_states_for(p.case_type, p.cont_type, [s], [t])
        data = learn.normalize(data) # comment out for non-linear kernel (Pavlidis)
        mean_fpr_best, mean_tpr_best, svm_clf_best, svm_v_best, pow_best = None, None, None, None, -1.0
        
        # multi-process grid search for best C
        results = []
        for c in p.c_grid:
            result = process_pool.apply_async(learn.svm_cross_val_ROC, (data, state, p.K, s, t, False, c))
            results.append((c,result))
        
        # wait for results
        for i, (c, result) in enumerate(results):
            mean_fpr, mean_tpr, svm_clf, svm_v = result.get()
            
            sys.stdout.write("(c=%g, p=%g) " % (c, mean_tpr[5]))
            if(mean_tpr[5] > pow_best):
                mean_fpr_best, mean_tpr_best, svm_clf_best, svm_v_best, pow_best = mean_fpr, mean_tpr, svm_clf, svm_v, mean_tpr[5]
            
            mean_fpr, mean_tpr, svm_clf, results[i] = None, None, None, None # clean up
        
        # save svm & power
        spec_svms[s,t] = svm_clf_best
        st2pow[s,t] = pow_best
        
        # plot ROC
        print ""
        learn.plot_roc(mean_fpr_best, mean_tpr_best, svm_clf_best.coef_[0,:], p.K, s, t) # comment for non-linear kernel
        mean_fpr_best, mean_tpr_best, svm_clf_best = None, None, None # clean up
        
        # save SVM vector for clustering/heatmap
        label = r"$\mathbf{%g \, %i \, (%.2f)}$" % (s, t, st2pow[s,t])
        labels.append(label)
        vects.append(svm_v_best)
                
        # report & label
        print "%g\t%i\t%g" % (s,t,st2pow[s,t]) 
    
    # write power to file
    write_power_to_file('power_SVM_specific.txt', st2pow)
    
    # compute SVM similarity & visualize (only linear kernel)
    # import clust_svm
    # metric = 'cosine' # 'correlation', 'euclidean'  
    # clust_svm.clust_svm(vects, labels, metric, 'SVM_%s_hclust.pdf' % metric, "SVM_%s_heatmap.pdf" % metric, None)
    
    # pickle spec_svms
    with open(spec_svm_file_to_write, mode='wb') as fh: 
        pickle.dump(spec_svms, fh)
    
################################## Pairwise SVC #####################################
def pairwise_SVC():
    ''' Compute power for all (s,t)-specific models, across all pairwise (s,t) comparisons, generate a heatmap.
        The diagonal of the resulting power matrix (i.e., each model applied to its respective scenario) is what 
        results from specific_SVC().
    '''
    
    pow_mat = [] # pairwise power matrix
    
    for s,t in [(s,t) for s in p.selection for t in p.times]:
        # get power with test data (s,t) for all training datasets  
        test_st_2_pow, test_st_pow = learn.svc_search(datas, states, p.K, s, t)
        pow_mat.append(test_st_pow)
    
    # visualize power as heatmap
    # clust_svm.heat_map(np.array(pow_mat), 'power', 'power_heatmap.png', 'Reds', False, False)
    # clust_svm.heat_map(np.array(pow_mat), 'power', 'power_heatmap_log-color.png', 'Reds', True, False)

########################### dotprod_and_decison_func_of ############################
def dotprod_and_decison_func_of(meta_svm, s, t, regime):
    ''' Debug: show dot-product & decision-function of data from given (s,t),
        using the 'regime' SVM. Exits when done.
    '''
    
    b = meta_svm.trained_SVMs[regime].intercept_[0]
    for sim in range(300, p.last_sim):
        xi = SFS_vectors[p.case_type, s, t, sim]
        prob = meta_svm.trained_SVMs[regime].predict_proba( np.array( [xi / np.linalg.norm(xi)] )) 
        dot_wt_xi = np.dot(xi / np.linalg.norm(xi), meta_svm.trained_SVMs[regime].coef_[0,:])
        print "case: %i" % sim + "\t" + "dot(w^T,xi): %.2f" % dot_wt_xi + "\t" + "b: %.2f" % b + "\t" + "decision-func: %.2f" % (dot_wt_xi + b) + "\t" + "prob: %.2f" % prob[0,1] 
    
    for sim in range(300, p.last_sim):
        xi = SFS_vectors[p.cont_type, s, t, sim]
        prob = meta_svm.trained_SVMs[regime].predict_proba( np.array( [xi / np.linalg.norm(xi)] ))
        dot_wt_xi = np.dot(xi / np.linalg.norm(xi), meta_svm.trained_SVMs[regime].coef_[0,:])
        print "cont: %i" % sim + "\t" + "dot(w^T,xi): %.2f" % dot_wt_xi + "\t" + "b: %.2f" % b + "\t" + "decision-func: %.2f" % (dot_wt_xi + b) + "\t" + "prob: %.2f" % prob[0,1]
    
    sys.exit(1)

################################ write_power_to_file ###############################
def write_power_to_file(fname, pow):
    ''' write power at different (s,t) to file '''
    results = open(fname,'w')
    for s,t in [(s,t) for s in p.selection for t in p.times]:          
        results.write("%g\t%i\t%g\n" % (s,t,pow[s,t]))
    results.close()

################################## write_to_file ###################################
def write_to_file(fname, write):
    ''' write list of strings to file '''
    file = open(fname,'w')
    for s in write:            
            file.write(s + "\n")   
    file.close()

############################# data_mix_for_SVM_general #############################
def data_mix_for_SVM_general(first, last):
    ''' training datasets for general SVM (near- and post-fixation) 
    '''
    train_mix = {}
    
    # set simulation range
    if(first is None): first = p.first_sim
    if(last is  None): last  = p.last_sim
    
    reg = regimes.get_regimes()
    for i, regime_dict in enumerate(reg):
         train_mix[i,"data"], train_mix[i,"states"] = data_and_states_for_dict(p.case_type, p.cont_type, regime_dict, first, last)
    
    return train_mix

######################### data_mix_for_SVM_general_exclude #########################
def data_mix_for_SVM_general_exclude(s, t):
    ''' training datasets for general SVM (near- and post-fixation), 
        excluding data from given parameters 
    '''
    
    train_mix = {}
    
    reg = regimes.get_regimes() 
    for i, regime_dict in enumerate(reg):  
         if(s in regime_dict.keys() and t in regime_dict[s]):
            regime_dict[s] = [x for x in regime_dict[s] if x != t] # to-exclude in current epoch, remove
        
         train_mix[i,"data"], train_mix[i,"states"] = data_and_states_for_dict(p.case_type, p.cont_type, regime_dict, p.first_sim, p.last_sim)
             
    return train_mix 

############################### data_and_states_for ################################
def data_and_states_for(case, cont, s_list, t_list, first=None, last=None):
    
    # set simulation range
    if(first is None): first = p.first_sim
    if(last  is None): last  = p.last_sim
    
    data,state  = [],[] # sfs-vectors, states
    for s in s_list:
        for t in t_list:
            for sim in range(first, last):
                case_v = SFS_vectors[case, s, t, sim]
                ########## remove xp-fixed bin from sp ##########
                if(not "xp" in case): case_v = case_v[:-1]
                data.append(case_v)
                state.append(1)

                cont_v = SFS_vectors[cont, s, t, sim]
                ########## remove xp-fixed bin from sp ##########
                if(not "xp" in cont): cont_v = cont_v[:-1]
                data.append(cont_v)
                state.append(-1)
    
    return np.array(data), np.array(state)

############################## data_and_states_for #################################
def data_and_states_for_dict(case, cont, dict, first, last):
    ''' returns data & states for the given dict
        such that keys are 's' and values are 't' at the given 's'
    '''
    
    data,state = [],[]
    for s in dict.keys():
        for t in dict[s]:
            for sim in range(first, last):
                case_v = SFS_vectors[case, s, t, sim]
                ########## remove xp-fixed bin from sp ##########
                if(not "xp" in case): case_v = case_v[:-1]
                data.append(case_v)
                state.append(1)

                cont_v = SFS_vectors[cont, s, t, sim]
                ########## remove xp-fixed bin from sp ##########
                if(not "xp" in cont): cont_v = cont_v[:-1]
                data.append(cont_v)
                state.append(-1)
                
    return np.array(data), np.array(state)

####################################################################################
#################################### MAIN ##########################################
####################################################################################
def main():
    global p
    
    parser = argparse.ArgumentParser(description='Supervised Learning of selective sweeps from the Site Frequency Spectrum.')
    
    # arguments
    parser.add_argument('outdir', help='name of output directory')
    
    # options
    parser.add_argument('-l', '--last'    , type=int, help='last simulation (number) to use for training & testing')
    parser.add_argument('-d', '--data'    , type=str, help='pre-processed SFS vectors, cPickled (.pck, FULL PATH)')
    parser.add_argument('-s', '--specific', type=str, help='pre-trained specific SVM model, cPickled (.pck, FULL PATH)')
    parser.add_argument('-g', '--general' , type=str, help='pre-trained general SVM model, cPickled (.pck, FULL PATH)')
    
    args = parser.parse_args()
    
    # prep
    if(args.last != None): p.last_sim = args.last
    os.mkdir(args.outdir)
    os.chdir(args.outdir)
    
    go(args.data, args.general, args.specific)
    print ""
    
if __name__ == '__main__':
    main()
