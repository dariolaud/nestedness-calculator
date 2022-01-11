import numpy as np


# Degree (rows)
def row_deg(M):

    '''
    This functions takes a bidimensional matrix (np.array) and returns a numpy array whose elements are the row degrees
    '''
    
    dim = M.shape
    Rdeg = np.array([sum(M[i]) for i in range(dim[0])])
    return Rdeg
   
   
# Degree (columns)
def col_deg(M):

    '''
    This functions takes a bidimensional matrix (np.array) and returns a numpy array whose elements are the column degrees
    '''
    
    dim = M.shape
    Cdeg = np.array([sum(M[:, i]) for i in range(dim[1])])
    return Cdeg
   
  
# -----------------------------------------------------------
 
 
# Degree of the lowest-degree neighbor of row-nodes
# min_{alpha : M_{i,alpha}=1} {k_{alpha}}
def Kb_min_i(M):

    '''
    This functions takes a bidimensional matrix (np.array) and returns a vector whose elements are the degrees of the lowest-degree neighbor of each row-node
    '''
    
    k_min = np.zeros(M.shape[1])
    K_MIN = np.zeros(M.shape[0]) # Vector whose elements are the degrees of the lowest-degree neighbor of each row-node
    Col_deg = col_deg(M)
    
    for i in range(M.shape[0]):
        for a in range(M.shape[1]):
            if M[i,a]>0: # condition that each column-node must fulfill to be considered as a possible candidate for the minimum
                k_min[a]=Col_deg[a] # each column node is a candidate
            else:
                k_min[a]=M.shape[0] # if node a does not interact with node i, then it cannot be counted for the minimum and thus it is assigned the maximum possible value (i.e. the number of rows)
        K_MIN[i] = np.amin(k_min)
        
    return K_MIN
    

# Degree of the largest-degree non-neighbor of row-nodes
# max_{alpha : M_{i,alpha}=0} {k_{alpha}}
def Kb_max_i(M):

    '''
    This functions takes a bidimensional matrix (np.array) and returns a vector whose elements are the degrees of the largest-degree non-neighbor of each row-node
    '''
    
    k_max = np.zeros(M.shape[1])
    K_MAX = np.zeros(M.shape[0]) # Vector whose elements are the degrees of the largest-degree non-neighbor of each row-node
    Col_deg = col_deg(M)
    
    for i in range(M.shape[0]):
        for a in range(M.shape[1]):
            if M[i,a]==0: # condition that each column-node must fulfill to be considered as a possible candidate for the maximum
                k_max[a]=Col_deg[a]
            else:
                k_max[a]=0 # if node a does not interact with node i, then it cannot be counted for the maximum and thus it is assigned the minimum possible value (i.e. 0)
        K_MAX[i] = np.amax(k_max)
        
    return K_MAX
    
