
''' Parameter settings for SFselect training.
    Models of the scaled SFS for classifying neutral evolution from selective sweeps.
'''

####################################################################################
#################################### General #######################################
####################################################################################

# prefix of path to simulation data dirs
sim_dir_pref = "/home/rronen/Documents/selection/data/sim_500_2.4e-07" 

# first & last simulation, used by reader to construct input file names
first_sim, last_sim = 0, 500

# case & control SFS vector types
case_type = "case_xpSFS" # can be: case,  case_xpSFS 
cont_type = "cont_xpSFS" # can be: cont1, cont_xpSFS 

# switch for learning from demographic simulations
demographic = False

# default name for data (SFS vectors) file
data_file = "sfs_vectors.pck"

# treatment of fixed mutations
ignore_xp_fixed  = True  # ignore fixed SNPs if fixed in both populations, otherwise keep
ignore_all_fixed = False # ignore fixed SNPs, used for strict theta purposes

# maximal number of frequency bins for binned SFS. This is exact number of bins 
# unless there are fewer haplotypes in the input sample, which shouldn't happen
max_bins      = 10 # default: 10 (also tried 8,16,20)
max_bins_case = 7  # default: 7  (also tried 6, 9,11)
max_bins_cont = 7  # default: 7  (also tried 6, 9,11)

# times points post selection
times  = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
times += [i for i in range(600,4001,100)]

# selection coefficients
selection = [0.005, 0.01, 0.02, 0.04, 0.08]

# starting frequency of beneficial allele
# ignore, here for complience with soft sweep file names
start_f = [0.0]

# cross-validation fold
K = 20

# classification error term
c_grid = [0.1] # [0.01, 0.1, 1.0]

# effective pop size N_e (haplotypes)
N = 2000

# site frequency folding, as in take min(f, 1-f)
# WARNING: not properly tested, use with caution
fold_freq = False 

####################################################################################
################################## for soft sweep ##################################
####################################################################################

# sim_dir_pref = "../data/sim_soft_500_2.4e-07" # soft sweep
# selection = [0.05]
# start_f = [0.5] # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
# c_grid = [1.0]

####################################################################################
############################ for demographic scenario ##############################
####################################################################################

#times = [60,80,100,150,200]
#selection = [0.20]
#sim_dir_pref = "sim_500_demographic"
#demographic = True

####################################################################################
############################ for mean scaled SFS plots #############################
####################################################################################

#selection = [0.08]
#times = [150, 250, 1000,2000]
#max_bins = 130
