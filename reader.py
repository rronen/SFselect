import sys, os, math, copy, cPickle
import params as p
import numpy as np

SFS_vectors = {} 
overwrote_pos = 0
binEdges = None

####################################################################################
def rescaled_binned_SFS(data_file):
    ''' reads data from files & returns a dict with scaled binned SFS vectors '''
    global SFS_vectors
    
    # set up
    bins, bins_case, bins_cont = None, None, None
    
    # report
    if(data_file is None):
        print "\nReading simulated population-sample data & generating binned scaled-SFS:\n"
        if(p.ignore_all_fixed):
            print "\tIgnoring fixed variants.\n"
        elif(p.ignore_xp_fixed):
            print "\tIgnoring variants fixed in all populations.\n"

        ##### read data #####
        for s in p.selection:

            for f in p.start_f:
                
                sim_dir = p.sim_dir_pref + "_s%.3f_f%.2f" % (s,f)
                
                print "\treading simulated data from %s" % sim_dir
                
                for t in p.times:

                    # for each (f,s,t) compute the scaled SFS, re-scaled to [0,1], of case, cont1, and cont2
                    
                    for sim in range(p.first_sim, p.last_sim):
                        # read sample frequencies
                        nhaps_case,  freq_case  = read_freq("%s/tests/%i.pop.case.samp.t%i.freq"  % (sim_dir, sim, t))
                        nhaps_cont1, freq_cont1 = read_freq("%s/tests/%i.pop.cont.samp1.t%i.freq" % (sim_dir, sim, t))
                        nhaps_cont2, freq_cont2 = read_freq("%s/tests/%i.pop.cont.samp2.t%i.freq" % (sim_dir, sim, t))
                        
                        # discard variants fixed in all populations
                        # discard_xp_fixed_all(freq_case, freq_cont1, freq_cont2)
                        
                        # bin edges
                        bins      = make_bins(bins     , p.max_bins     , nhaps_case, nhaps_cont1, nhaps_cont2)
                        bins_case = make_bins(bins_case, p.max_bins_case, nhaps_case, nhaps_cont1, nhaps_cont2)
                        bins_cont = make_bins(bins_cont, p.max_bins_cont, nhaps_case, nhaps_cont1, nhaps_cont2)
                        
                        # scaled-SFS vectors
                        SFS_vectors["case",  s,t,sim] = bin_SFS(freq_case.values() , bins)
                        SFS_vectors["cont1", s,t,sim] = bin_SFS(freq_cont1.values(), bins)
                        SFS_vectors["cont2", s,t,sim] = bin_SFS(freq_cont2.values(), bins)
                        
                        # XP-SFS vectors
                        if(p.demographic):
                            # CEU/YRI demography
                            freq_case_clean, freq_cont2_clean = discard_xp_fixed_pair(freq_case, freq_cont2) 
                            SFS_vectors["case_xpSFS",s,t,sim] = bin_xp_SFS(freq_case_clean,  freq_cont2_clean, bins_case, bins_cont)
                        else:
                            # constant sized populations
                            freq_case_clean, freq_cont1_clean = discard_xp_fixed_pair(freq_case, freq_cont1) 
                            SFS_vectors["case_xpSFS",s,t,sim] = bin_xp_SFS(freq_case_clean,  freq_cont1_clean, bins_case, bins_cont)
                        
                        freq_cont1_clean, freq_cont2_clean = discard_xp_fixed_pair(freq_cont1, freq_cont2)
                        SFS_vectors["cont_xpSFS",s,t,sim] = bin_xp_SFS(freq_cont1_clean, freq_cont2_clean, bins_case, bins_cont)
                        
        
	# report done reading SFS vectors
        print "\n" + "Done! (overwrote pos %i times)" % overwrote_pos + "\n"
        
        # pickle for prosperity
        with open(p.data_file, mode='wb') as sfs_fh:
            cPickle.dump(SFS_vectors, sfs_fh)

    else:
        # read from pickled file
        sys.stdout.write("\nLoading cPickle data from: %s... " % data_file)
        with open(data_file,mode='rb') as sfs_fh: 
            SFS_vectors = cPickle.load(sfs_fh)
        print "done.\n" 
        
    return SFS_vectors

################################### read_freq ######################################
def read_freq(file):
    ''' read frequencies from file and return as: dict[pos]=f '''
    global overwrote_pos
    nhaps, freq = 0, {}
    
    File = open(file, 'r')
    for line in File:
        sline = line.rstrip().split()
        if(line.startswith("#")):
            nhaps = int(sline[1])
        else:
            pos_i, f, pos_d = int(sline[0]), float(sline[1]), float(sline[2])
            pos_use = pos_d # if pos_i used, equal consecutive positions overwritten
            
            # ignore fixed
            if( (p.ignore_all_fixed or p.fold_freq) and f == 1.0): continue
            
            if(pos_use in freq): overwrote_pos += 1 # count overwrites
            
            if( p.fold_freq ):
                freq[pos_use] = min(f, 1.0-f)
            else:
                freq[pos_use] = f
    
    File.close()

    return nhaps, freq

################################ discard_xp_fixed_all ###############################
def discard_xp_fixed_all(freq_case, freq_cont1, freq_cont2):
    ''' discard variants fixed in case & cont1 & cont2 '''
    
    if( (not p.ignore_all_fixed) and p.ignore_xp_fixed ):
        # remove variants fixed in case & cont1 & cont2
        for pos in freq_case.keys():
            if(freq_case[pos] == 1.0 and pos in freq_cont1 and freq_cont1[pos] == 1.0 and pos in freq_cont2 and freq_cont2[pos] == 1.0):
                del(freq_case[pos] )
                del(freq_cont1[pos])
                del(freq_cont2[pos])
                
############################### discard_xp_fixed_pair ###############################
def discard_xp_fixed_pair(freq_case, freq_cont):
    ''' return copies of {case,cont} frequency dictionaries, 
        with variants that are fixed in both populations removed '''
    
    # copies, will be changed and returned
    freq_case_clean = copy.deepcopy(freq_case)
    freq_cont_clean = copy.deepcopy(freq_cont)
    
    # discard variants fixed in both populations
    if( (not p.ignore_all_fixed) and p.ignore_xp_fixed ):
        # remove variants fixed in case & cont
        for pos in freq_case.keys():
            if(freq_case[pos] == 1.0 and pos in freq_cont and freq_cont[pos] == 1.0):
                del(freq_case_clean[pos])
                del(freq_cont_clean[pos])
    
    return freq_case_clean, freq_cont_clean

##################################### bin_SFS ######################################
def bin_SFS(freq, bins):
    global binEdges
    
    v, binEdges = np.histogram(freq, bins)
    centers = 0.5*(binEdges[1:]+binEdges[:-1])
    v = v*centers
    # v = np.array(v, dtype=float) # if not weighted, make float
    return v

#################################### make_bins #####################################
def make_bins(bins, max_bins, n_case, n_cont1, n_cont2):
    if(bins is None):
        nbins = min(max_bins, n_case, n_cont1, n_cont2)
        if(p.fold_freq):
            bins  = lrange( 0.0, 0.5, 0.5/float(nbins) )
        else:
            bins  = lrange( 0.0, 1.0, 1.0/float(nbins) )
    return bins

##################################### lrange #######################################
def lrange(start, stop, step):
    
    # basic bins in [0,1]
    l = list( np.arange(start, stop, step) )
    
    # special bin for fixed variants
    if(not p.fold_freq):
         l.append(np.nextafter(1,0)) # only if not folding frequencies
    
    l.append(stop)
    
    return l

################################### bin_xp_SFS #####################################
def bin_xp_SFS(freq_case, freq_cont, bins_case, bins_cont):
    ''' bins and re-scales a 2D (cross-population) SFS, returns a linearized version '''
    
    union_case, union_cont =  union_snp_freq_dicts(freq_case, freq_cont)
    v, x_edges, y_edges = np.histogram2d(union_case.values(), union_cont.values(), np.array([bins_case, bins_cont]) )

    # bin centers
    x_centers = 0.5*(x_edges[1:]+x_edges[:-1])
    y_centers = 0.5*(y_edges[1:]+y_edges[:-1])
    
    # scaling
    xpscale = np.outer(x_centers, y_centers.T)
    v = v*xpscale
    
    return v.reshape(-1)

############################### union_snp_freq_dicts ###############################
def union_snp_freq_dicts(freq_case, freq_cont):
    ''' unite case & control dict keys '''
    
    union_case, union_cont = {}, {}
    
    for pos,f_case in freq_case.iteritems():
        # add all case frequencies to case-union
        union_case[pos] = f_case
        
        # add case-specific frequencies to cont-union
        if(not pos in freq_cont):
            union_cont[pos] = 0.0
        
    for pos,f_cont in freq_cont.iteritems():
        # add all cont frequencies to cont-union 
        union_cont[pos] = f_cont
        
        # add cont-specific frequencies to case-union
        if(not pos in freq_case):
            union_case[pos] = 0.0
    
    # sanity check
    assert (len(union_case.keys()) == len(union_cont.keys())), "sfs.py::union_snp_freq_dicts::key union error"
    
    return union_case, union_cont

