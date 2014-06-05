import sys, os

########################### get_regimes #############################
def get_regimes():
    e0,e1 = {},{}
    
    # near fixation block
    e0[0.02]  = [400, 450] + [i for i in range(500,1001,100)]
    e0[0.04]  = [200,250,300,350,400,450,500,600]                 
    e0[0.08]  = [100,150,200,250,300]                             
    
    # post fixation block
    e1[0.02]  = [i for i in range(1100,4001,100)]
    e1[0.04]  = [i for i in range(700, 4001,100)]                 
    e1[0.08]  = [350,400,450] + [i for i in range(500, 4001,100)]

    return e0, e1

######################## get_regimes_three ##########################
def get_regimes_three():
    e0,e1,e2 = {},{},{}
    
    # near fixation block
    e0[0.02]  = [400, 450] + [i for i in range(500,1001,100)]
    e0[0.04]  = [200,250,300,350,400,450,500,600]
    e0[0.08]  = [100,150,200,250,300]
    
    # post fixation block
    e1[0.02]  = [i for i in range(1100,2801,100)]
    e1[0.04]  = [i for i in range(700, 2201,100)]
    e1[0.08]  = [350,400,450] + [i for i in range(500, 1501,100)]

    # late block
    e2[0.02]  = [i for i in range(2900, 4001,100)]
    e2[0.04]  = [i for i in range(2300, 4001,100)]
    e2[0.08]  = [i for i in range(1600, 4001,100)]

    return e0, e1, e2

############################ regime_of ##############################
def regime_of(s,t):
    """ returns the index of the regimes which s,t belong to, or -1 if belongs to none """
    for i, regime_dict in enumerate(get_regimes()):
        if(s in regime_dict.keys() and t in regime_dict[s]):
            return i
    
    return -1
