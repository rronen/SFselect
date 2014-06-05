#!/usr/bin/env python

import sys, argparse
import numpy as np
import cPickle as pickle
from collections import defaultdict
import bisect, operator

#  SFselect.py
#
#  This program takes
#  (1) a trained model (SVM) of the site frequency spectrum.
#  (2) variant frequencies along a gemomic segment.
#  
#  It applies the given model to the frequency data and outputs class probabilities. Classes are
#  genmic segments evolving either neutrally, or under a 'hard sweep' model of positive selection.
#  The model is applied to a sliding window of constant size along the given segment. 

###############################################################################
############################ Parameter Defaults ###############################
###############################################################################

# defaults for internal parameters

nbins       = 10     # single-population non-fixed bins
nbins_case  = 7      # cross population non-fixed bins for control 
nbins_cont  = 7      # cross population non-fixed bins for case
GEN_SVM     = True   # internal indicator for general/specific
XP_SFS      = False  # internal indicator for SP/XP
flt_fly     = True   # fly genome      : skip chromosomes other than 2L,2R, 3L, 3R, X
flt_draft   = False  # draft assemblies: skip contigs without "chromosome" in name (e.g. chlamy)
bins        = []     # bins for scaled SFS
test_scores = {}     # accumulates results

# optional parameters, overwritten from cmd

specific_s = None # selection  pressure  of specific SVM
specific_t = None # time under selection of specific SVM  
gt_to_1    = None # rounds higher freq to 1.0 [0.99]
lt_to_0    = None # rounds lower  freq to 0.0 [0.00]
w_size     = None # sliding window size in bp [50000] 
step       = None # sliding window step size in bp [2000] 
min_snps   = None # skip windows with fewer variants [10]
out_pref   = ""   # prefix for output file
CHR        = ""   # chromosome name for single chromosome runs (e.g. human)

###############################################################################
################################### GO ########################################
###############################################################################
def sfselect(svm_file, case_file, cont_file):
    global bins, bins_case, bins_cont
    
    # prep
    res_file_pos  = out_pref + "sfselect.srt.pos"
    res_file_stat = out_pref + "sfselect.srt.stat"
    
    bins      = lrange( 0.0, 1.0, 1.0/float(nbins)      )
    bins_case = lrange( 0.0, 1.0, 1.0/float(nbins_case) )
    bins_cont = lrange( 0.0, 1.0, 1.0/float(nbins_cont) )
            
    # open & header position-sorted results
    out_pos_srt = open_w_header(res_file_pos)
    
    # read & load pickled SVM file
    svm = read_pck_SVM(svm_file)
    
    # read population (& control) frequencies
    case_freqs = read_freq_file(case_file)
    if(XP_SFS): cont_freqs = read_freq_file(cont_file)
    
    # traverse chromosomes
    for chr in sorted(case_freqs.keys()):
        
        # only {2L,2R, 3L, 3R, X} if fly filter
        if(flt_fly and not chr in ["2L","2R","3L","3R","X"]): continue
        
        # only chromosomes if draft filter
        if(flt_draft and not "chromosome" in chr): continue

        # prep
        case_freqs_chr = case_freqs[chr]
        if(XP_SFS): 
            cont_freqs_chr = cont_freqs[chr]
            removed = remove_xp_fixed(case_freqs_chr, cont_freqs_chr)
        
        # verify there is data to test in current chromosome
        if( len(case_freqs_chr) == 0 or (XP_SFS and len(cont_freqs_chr) == 0) ): continue
        
        # get {min, max} SNP positions for scan & report
        case_pos_srt = sorted(case_freqs_chr.keys())
        if(XP_SFS):
            cont_pos_srt = sorted(cont_freqs_chr.keys()) 
            min_pos, max_pos = min(case_pos_srt[0], cont_pos_srt[0]), max(case_pos_srt[-1], cont_pos_srt[-1])
            print "\n" + "Scanning chromosome {} (from position {:,} to {:,}, ignoring {:,} xp-fixed SNPs)".format(chr, min_pos, max_pos, removed)
        else:
            min_pos, max_pos = case_pos_srt[0], case_pos_srt[-1]
            print "\n" + "Scanning chromosome {} (from position {:,} to {:,})".format(chr, min_pos, max_pos)
        
        # perform scan
        n_win,skp_win = 0, 0
        for start in range(min_pos, max_pos, step):
            
            # get case window frequencies, skip if needed
            sweep_pop_freqs_w = freqs_for_window(case_freqs_chr, case_pos_srt, start, start + w_size)
            if(len(sweep_pop_freqs_w) < min_snps):
               skp_win += 1
               continue
           
            # get control window frequencies, skip if needed
            if(XP_SFS):
                neut_pop_freqs_w = freqs_for_window(cont_freqs_chr, cont_pos_srt, start, start + w_size)
                if(len(neut_pop_freqs_w) < min_snps): 
                    skp_win += 1
                    continue
            
            # get normalized, binned, extended SFS or XP-SFS
            if(XP_SFS): 
                sfs_v = bin_scale_norm_xp(sweep_pop_freqs_w, neut_pop_freqs_w)
            else:
                sfs_v = bin_scale_norm(sweep_pop_freqs_w)
            
            # SFselect test
            if(GEN_SVM):
                w_prob, svm2prob = svm.predict_proba( np.array([sfs_v]) )
            else:
                w_prob           = svm.predict_proba( np.array([sfs_v]) )
            
            # log transform (ranges in ~ [0.01, 4.6])
            score_w = -np.log(1 - w_prob[0,1])
            
            # regime probabilities
            if(GEN_SVM): 
                near_p, post_p = svm2prob[0], svm2prob[1]
            else:
                prob = w_prob[0,1]
            
            # write & store
            if(XP_SFS):
                if(GEN_SVM):
                    res = "%s\t%i\t%i\t%i\t%i\t%g\t%g\t%g\n" % (chr, start, start + w_size, len(sweep_pop_freqs_w), 
                                                                len(neut_pop_freqs_w), score_w, near_p, post_p)
                else:
                    res = "%s\t%i\t%i\t%i\t%i\t%g\t%g\n" % (chr, start, start + w_size, len(sweep_pop_freqs_w), 
                                                            len(neut_pop_freqs_w), score_w, prob)
            else:
                if(GEN_SVM):
                    res = "%s\t%i\t%i\t%i\t%g\t%g\t%g\n" % (chr, start, start + w_size, len(sweep_pop_freqs_w), score_w, near_p, post_p)
                else:
                    res = "%s\t%i\t%i\t%i\t%g\t%g\n" % (chr, start, start + w_size, len(sweep_pop_freqs_w), score_w, prob)
            
            test_scores[res] = score_w
            out_pos_srt.write(res)
            n_win += 1
        
        print "Scanned {:,} windows (skipped {:,})".format(n_win, skp_win)
    
    out_pos_srt.close()
    
    # open & header stat-sorted results
    out_stat_srt = open_w_header(res_file_stat)
    
    # write stat-sorted results
    for to_write, score in sorted(test_scores.iteritems(), key=operator.itemgetter(1), reverse=True):
        out_stat_srt.write(to_write)
    out_stat_srt.close()
    print "\n" + "Done (output in {}, {})".format(res_file_pos, res_file_stat) + "\n"

########################### open_w_header ###########################
def open_w_header(fname):
    outf = open(fname, 'w')
    if(XP_SFS):
        if(GEN_SVM):
            outf.write("#" + "chr" + "\t" + "start" + "\t" + "stop" + "\t" + "#SNPs-sweep" + "\t" 
                           + "#SNPs-neutral" + "\t" + "XP-SFselect" + "\t" + "near-fix-prob" + "\t" 
                           + "post-fix-prob" + "\n")
        else:
            outf.write("#" + "chr" + "\t" + "start" + "\t" + "stop" + "\t" + "#SNPs-sweep" + "\t" 
                           + "#SNPs-neutral" + "\t" + "specific-SVM" + "\t" + "SVM-prob" + "\n")
    else:
        if(GEN_SVM):
            outf.write("#" + "chr" + "\t" + "start" + "\t" + "stop" + "\t" + "#SNPs-sweep" + "\t" 
                           + "SFselect" + "\t" + "near-fix-prob" + "\t" + "post-fix-prob" + "\n")
        else:
            outf.write("#" + "chr" + "\t" + "start" + "\t" + "stop" + "\t" + "#SNPs-sweep" + "\t" 
                           + "specific-SVM" + "\t" + "SVM-prob" + "\n")
            
    return outf

########################### read_pck_SVM ############################
def read_pck_SVM(svm_file):
    print "\n" + "Loading SVM file: %s" % svm_file
    
    with open(svm_file,mode='rb') as fh:
        svm = pickle.load(fh) # general SVM
        
        # if needed, overwrite with specific SVM
        if(isinstance(svm, dict)): 
            if(specific_s != None and specific_t != None):
                svm = svm[specific_s, specific_t]
            else:
                print "\n\t" + "Error::specific SVM input using general argument switch. Quitting...\n"
                sys.exit(1)
    return svm

########################## bin_scale_norm ###########################
def bin_scale_norm(freqs):
    ''' returns the binned, scaled, and normalized SFS '''
    
    v, binEdges = np.histogram(freqs.values(), bins)
    centers = 0.5*(binEdges[1:]+binEdges[:-1])
    v = v*centers

    return v / np.linalg.norm(v) 

######################## bin_scale_norm_xp ##########################
def bin_scale_norm_xp(freqs_case, freqs_cont):
    ''' returns the binned, scales, and normalized XP-SFS
        note: in linearized form 
    '''
    
    union_case, union_cont =  union_snp_freq_dicts(freqs_case, freqs_cont)
    v, x_edges, y_edges = np.histogram2d(union_case.values(), union_cont.values(), np.array([bins_case, bins_cont]) )

    # bin centers
    x_centers = 0.5*(x_edges[1:]+x_edges[:-1])
    y_centers = 0.5*(y_edges[1:]+y_edges[:-1])
    
    # scaling
    xpscale = np.outer(x_centers, y_centers.T)
    v = v*xpscale
    
    lin_v = v.reshape(-1)
    
    return lin_v / np.linalg.norm(lin_v)

####################### union_snp_freq_dicts ########################
def union_snp_freq_dicts(freq_sweep, freq_neut):
    ''' unite dict keys to be the union of sweep & neutral '''
    union_sweep, union_neut = {}, {}
    
    for pos,f_case in freq_sweep.iteritems():
        # add all case frequencies to sweep-union
        union_sweep[pos] = f_case
        
        # add case-specific frequencies to neutral-union
        if(not pos in freq_neut):
            union_neut[pos] = 0.0
        
    for pos,f_cont in freq_neut.iteritems():
        # add all cont frequencies to neutral-union 
        union_neut[pos] = f_cont
        
        # add cont-specific frequencies to sweep-union
        if(not pos in freq_sweep):
            union_sweep[pos] = 0.0
    
    # sanity check
    assert (len(union_sweep.keys()) == len(union_neut.keys())), "Error::union_snp_freq_dicts::key union error"
    return union_sweep, union_neut

#####################################################################
def remove_xp_fixed(freqs1, freqs2):
    ''' removes variants fixed in both sweep & neutral populations
        returns number of variants removed
    '''
    removed_c = 0
    for pos in freqs1.keys():
        if(freqs1[pos] == 1.0 and pos in freqs2 and freqs2[pos] == 1.0):
            del(freqs1[pos])
            del(freqs2[pos])
            removed_c += 1
    
    return removed_c

#####################################################################
def freqs_for_window(freqs, pos_srt, start, end):
    ''' returns a dict of positions -> frequencies for the given window '''
    
    freqs_w = {}
    curr_i = bisect.bisect_left(pos_srt, start)
    if(curr_i == len(pos_srt)): return freqs_w # start is after the last SNP
    
    curr_pos = pos_srt[curr_i]
    while(curr_pos <= end):
        freqs_w[curr_pos] = freqs[curr_pos]
        curr_i += 1
        if(curr_i == len(pos_srt)): break
        curr_pos = pos_srt[curr_i]
        
    return freqs_w
        
#####################################################################
def read_freq_file(file):
    ''' returns dict of dicts, where keys are (1) chromosomes (2) positions '''
    tri_c, tri_f = 0, 0
    
    freqs = defaultdict(dict)
    File = open(file, 'r')
    for line in File:
        sline = line.rstrip().split()
        if(line.startswith("Chr") or line.startswith("#")): 
            continue
        else:
            if(CHR != ""):
                pos, f = int(sline[0]), float(sline[1])
                chr = CHR # from command line
            else:
                chr, pos, f = sline[0], int(sline[1]), float(sline[2])
        
        # round to 1.0
        if(f > gt_to_1): f = 1.0

        # round to 0.0
        if(f < lt_to_0): continue

        if(pos in freqs[chr]):
            # site exists, implies triallelic
            # try adding frequency at consecutive position
            if(pos+1 in freqs[chr]):
                tri_f += 1 # failed, give up
            else:
                freqs[chr][pos+1] = f
                tri_c += 1
        else:
            # regular site, add frequency
            freqs[chr][pos]   = f
    
    File.close()
    if(tri_f > 0): print "NOTICE::added %i (failed to add %i) tri (or more) allelic sites while reading %s" % (tri_c,tri_f, file)
    return freqs

#####################################################################
def lrange(start, stop, step):
    ''' returns bin edges including a last bin exclusive to fixed variants '''
    
    # basic bins in [0,1]
    l = list( np.arange(start, stop, step) )
        
    # special bin for fixed variants
    l.append(np.nextafter(1,0)) 
    
    l.append(stop)
    
    return l

#####################################################################
############################# MAIN ##################################
#####################################################################
def main():
    global XP_SFS, GEN_SVM, CHR, w_size, step, out_pref, specific_s, specific_t, gt_to_1, lt_to_0, min_snps
    
    # command line parser
    parser = argparse.ArgumentParser(description='Supervised Learning from the Site Frequency Spectrum using SVMs.')
    
    # input
    parser.add_argument('pop',   help='population sample frequencies')
    
    # optional arguments
    parser.add_argument('--npop', help='neutral population sample frequencies (for XP-SFselect)', required=False)
    parser.add_argument('--spgSVM', type=str, help='single population (SP) general SVM', required=False)
    parser.add_argument('--xpgSVM', type=str, help='cross population (XP) general SVM', required=False)
    parser.add_argument('--spsSVM', type=str, help='single population (SP) specific SVMs (requires -s, -t)', required=False)
    parser.add_argument('--xpsSVM', type=str, help='cross population (XP) specific SVMs (requires -s, -t)', required=False)
    parser.add_argument('-s',   type=float, help='selection  pressure , one of {0.01,0.02,0.04,0.08}', required=False)
    parser.add_argument('-t',   type=int, help='time under selection, one of {100, 200, ..., 4000}', required=False)
    parser.add_argument('--wsize', type=int, help='size (bp) of sliding window [50000]', required=False, default=50000)
    parser.add_argument('--step',  type=int, help='step size (bp) for sliding window [2000]', required=False, default=2000)
    parser.add_argument('--mins',  type=int, help='skip windows with fewer SNPs [10]', required=False, default=10)
    parser.add_argument('--to1',  type=float, help='round higher frequencies to 1 [0.99]', required=False, default=0.99)
    parser.add_argument('--to0',  type=float, help='round lower  frequencies to 0 [0.00]', required=False, default=0.00)
    parser.add_argument('--out',   type=str, help='prefix for output files (e.g., chr1)', required=False)
    parser.add_argument('--chr',   type=str, help='chromosome name (required for single chromosome genomes, expects two column input)', required=False)
    
    args = parser.parse_args()
    
    # determine {SP, XP} and {general, specific}    
    svm_c = 0
    if(args.xpgSVM != None):
        svm = args.xpgSVM
        XP_SFS = True
        svm_c = svm_c + 1
    if(args.xpsSVM != None):
        svm = args.xpsSVM
        XP_SFS = True
        GEN_SVM = False
        svm_c = svm_c + 1
    if(args.spgSVM != None):
        svm = args.spgSVM
        svm_c = svm_c + 1
    if(args.spsSVM != None):
        svm = args.spsSVM
        GEN_SVM = False
        svm_c = svm_c + 1
    
    # make sure exactly 1 SVM given 
    if(svm_c != 1):
        print "\n\t" + "Error::exactly one of {--spgSVM, --xpgSVM, --spsSVM, --xpsSVM} should be specified. Quitting...\n"
        sys.exit(1)
    
    # make sure neutral frequencies file given if XP_SFS
    if(XP_SFS and args.npop == None):
        print "\n\t" + "Error::neutral population frequencies must be given for cross-population (XP) SVM. Quitting...\n"
        sys.exit(1)
    
    # make sure (s,t) given and sane if specific SVM
    if(args.spsSVM != None or args.xpsSVM != None):
        if(args.s == None or args.t == None):
            print "\n\t" + "Error::(s,t) must be given when using specific SVM. Quitting...\n"
            sys.exit(1)
        elif(not args.s in [0.01,0.02,0.04,0.08]):
            print "\n\t" + "Error:: (s) must be in {0.01,0.02,0.04,0.08}. Quitting...\n"
            sys.exit(1)
        elif(not args.t in [x for x in range(100,4001,100)]):
            print "\n\t" + "Error:: (t) must be in {100,200,...,4000}. Quitting...\n"
            sys.exit(1)
        else:
            specific_s = args.s
            specific_t = args.t
    
    # get optional arguments
    gt_to_1 = args.to1
    lt_to_0 = args.to0
    min_snps = args.mins
    w_size = args.wsize
    step = args.step
    if(args.chr != None): CHR = args.chr
    if(args.out != None): out_pref = args.out + '.'
    
    # more sanity checks
    if(gt_to_1 < 0 or gt_to_1 > 1):
        print "\n\t" + "Error:: 'to1' argument must be in [0,1]. Quitting...\n"
        sys.exit(1)
    
    if(lt_to_0 < 0 or lt_to_0 > 1):
        print "\n\t" + "Error:: 'to0' argument must be in [0,1]. Quitting...\n"
        sys.exit(1)
        
    if(min_snps < 0):
        print "\n\t" + "Error:: 'mins' argument must be positive. Quitting...\n"
        sys.exit(1)
    
    # run genomic scan
    sfselect(svm, args.pop, args.npop)
    
if __name__ == '__main__':
    main()
