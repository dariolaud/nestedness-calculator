import numpy as np
import deg
import statistics
import math


'''
Implemented metrics:

- Departures
- Discrepancies (two versions)
- Normalized Discrepancies
- Spectral radius
- Normalized Spectral Radius
- Unexpected absences and presences
- Wright and Reeves (for rows and columns)
- Standardized Wright and Reeves
- JDM
- Standardized JDM
- Tau Temperature (Manhattan distance)
- NODF (homogeneous and heterogeneous normalizations)
- SNODF (homogeneous and heterogeneous normalizations)
- Global Nestedness (homogeneous and heterogeneous normalizations)
'''


def Departures(M): # Introduced in [1]

    '''
    For an ORDERED MATRIX, the number of departures is defined as the number of times the absence of a node is followed by its presence in the neighborhood of the next lower-degree node. In other words, the number of departures is the number of times the ith row does not interact with colomun  𝛼  but the  (𝑖+1) th row does.
    The bigger this index, the less nested the network.
    
    This functions takes a bidimensional matrix (np.array) and returns the number of Departures
    '''
    
    M_p = M[np.argsort(-M.sum(axis=1))]
    M_o = M_p[:, np.argsort(-M_p.sum(axis=0))] # Ordered matrix
    r,c = M_o.shape
    D=0
    
    for i in range(r-1):
        for j in range(c):
            if (M_o[i,j]<M_o[i+1,j]): # If row i does not interact with column j but row (i+1) does
                D=D+1
                
    return D


# --------------------------------------------------------------------------------

def Discrepancies(M): # Introduced in [2]

    '''
    From the original matrix, one constructs a corresponding perfectly nested matrix by shifting, for each row, all its filled elements to the left. The elements of the observed matrix that do not match the corresponding ones for the perfectly nested matrix are interpreted as 'discrepancies'.
    The bigger this index, the less nested the network.
    
    This functions takes a bidimensional matrix (np.array) and returns the Discrepancies
    '''
    
    M_disp = np.zeros_like(M) # Initialize the corresponding perfectly nested matrix
    Row_deg = deg.row_deg(M)
    
    for i in range(len(Row_deg)):
        M_disp[i,0:Row_deg[i]]=1 # The ones are moved left
        
    D = sum(sum((M-M_disp)==1)) # Matchings
        
    return D
   
   
  
def Discrepancies_alternative(M): # Alternative (but equivalent) computation of the Discrepancies
    
    '''
    Alternative but equivalent computation of the Discrepancies.
    
    This functions takes a bidimensional matrix (np.array) and returns the Discrepancies.
    '''
    
    M_disp = np.zeros_like(M) # Initialize the corresponding perfectly nested matrix
    Row_deg = deg.row_deg(M)
     
    for i in range(len(Row_deg)):
         M_disp[i,0:Row_deg[i]]=1 # The ones are moved left
         
    D = sum(M<M_disp)
    D =(sum(D))
         
    return D
     
   
   
def NormDiscrepancies(M): # Normalized version of the Discrepancies

    '''
    Normalized version of the Discrepancies, where the original index is divided by the number of presences.
    
    This functions takes a bidimensional matrix (np.array) and returns the Normalized Discrepancies, by calling the function Discrepancies1() and dividing the result by the number of presences
    '''
    
    f = M.sum() # Number of presences
    D = Discrepancies(M) # Obtain the discrepancies
    D1 = D/f # Normalization
    
    return D1


# --------------------------------------------------------------------------------


# Consider all the connected bipartite networks composed of S nodes and E edges. The network with the largest spectral radius is a perfectly nested network
# Consider all the connected bipartite networks composed of N row-nodes, M column-nodes, and E edges. The network with the largest spectral radius is a perfectly nested network
# If perfectly nested structures lead to the largest spectral radius for bipartite networks, the spectral radius  𝜌(𝐴)  can be used to quantify the degree of nestedness of a given network: the larger  𝜌(𝐴) , the more nested the network

def SpectralRadius(M): # Introduced in [3]

    '''
    The spectral radius of a square matrix is the largest absolute value of its eigenvalues.
    The larger this index, the more nested the network.
    
    This function takes a bidimensional matrix (np.array) and returns and returns its spectral radius
    '''
    
    r,c=M.shape
    s = r+c
    
    Adj = np.zeros((s,s))
    Adj[0:r, r:s]=M
    Adj = Adj+np.transpose(Adj) # Adjacency matrix
    
    M_eigenvalues = np.linalg.eigvals(Adj) # Eigenvalues
    Max_M = np.amax(abs(np.real(M_eigenvalues)))
    
    return Max_M
    
    

def Normalized_SpectralRadius(M):  # Normalized version of the Spectral Radius (divided by sqrt(L))

    '''
    Normalized version of the spectral radius, where the original index is divided by the square root of the number of precenses.
    
    This function takes a bidimensional matrix (np.array) and returns its normalized spectral radius.
    '''
    
    r,c=M.shape
    L = M.sum() # Number of presences
    s = r+c
    
    Adj = np.zeros((s,s))
    Adj[0:r, r:s]=M
    Adj = Adj+np.transpose(Adj) # Adjacency matrix
    
    M_eigenvalues = np.linalg.eigvals(Adj) # Eigenvalues
    Max_M = np.amax(abs(np.real(M_eigenvalues))) # Spectral Radius
    Max_Mn = Max_M/math.sqrt(L) # Normalization
    
    return Max_Mn
    
    
# --------------------------------------------------------------------------------
    
def UnexpectedAbsences(M): # Introduced in [4]
    
    '''
    The number of unexpected absences counts how many times a node 𝑖 is not a neighbor of a node 𝛼 that has a larger degree than its lowest-degree neighbor.
    The largest this index, the less nested the network.
    
    This function takes a bidimensional matrix (np.array) and returns the number of Unexpected absences.
    '''
    
    N0 = 0
    Col_deg = deg.col_deg(M) # k_alfa (array that contains the degree of the columns)
    Kmin = deg.Kb_min_i(M) # min_{alpha : M_{i,alpha}=1} {k_{alpha}}
    
    for i in range(M.shape[0]):
        for a in range(M.shape[1]):
            if (Col_deg[a]>Kmin[i]):
                N0 = N0 + 1 - M[i,a]
                
    return N0
    
    
    
def UnexpectedPresences(M):

    '''
    The number of unexpected presences counts how many times a node 𝑖 is a neighbor of a node 𝛼 that has a smaller degree than its largest-degree non-neighbor.
    The largest this index, the less nested the network.
    
    This function takes a bidimensional matrix (np.array) and returns the number of Unexpected presences.
    '''
    
    N1 = 0
    Col_deg = deg.col_deg(M) # k_alfa k_alfa (array that contains the degree of the columns)
    Kmax = deg.Kb_max_i(M) # max_{alpha : M_{i,alpha}=0} {k_{alpha}}
    
    for i in range(M.shape[0]):
        for a in range(M.shape[1]):
            if (Kmax[i]>Col_deg[a]):
                N1 = N1 + M[i,a]
                
    return N1
    
    
# --------------------------------------------------------------------------------

def WrightReeves(M): # Introduced in [5]

    '''
    In a perfectly nested matrix, if a species is found on an island, then it should be also found on richer islands. Therefore nestedness can be quantified by aggregating the overlaps between the neighborhoods of pairs of islands.
    This metrics become larger as the degree of nestedness increases.
    
    This function takes a bidimensional matrix (np.array) and returns the Wright-Reeves overlap indicator, computed for rows.
    '''
    
    r,c = M.shape
    k = np.zeros(r)
    
    for i in range(r):
        k[i] = sum(M[i,:])
    Nc = (1./2.)*sum(k*(k-1)) # N_c = (1/2)*sum_{i}(k_i*(k_i - 1))
    
    return Nc
    


def WrightReeves_inv(M): # The computation is carried out for columns instead of rows

    '''
    Complementary version of the original Wright and Reeves index, where the computation is carried out for columns instead of rows.
    
    This function takes a bidimensional matrix (np.array) and returns the Wright-Reeves overlap indicator, computed for columns.
    '''
    
    r,c = M.shape
    k = np.zeros(c)
    
    for i in range(c):
        k[i] = sum(M[:,i])
    Nc = (1./2.)*sum(k*(k-1)) # N_c = (1/2)*sum_{i}(k_i*(k_i - 1))
    
    return Nc
            
    

def WrightReeves_alternative(M): # Alternative implementation using mean values

    '''
    Alternative computation of the Wright and Reeves metrics (using mean values).
    
    This function takes a bidimensional matrix (np.array) and returns the Wright-Reeves overlap indicator, computed for rows.
    '''
    
    r,c = M.shape
    k = np.zeros(r)
    
    for i in range(r):
        k[i] = sum(M[i,:])
        
    mean = statistics.mean(k)
    mean2 = statistics.mean(k*k)
    Nc = (r/2.)*(mean2-mean)
    
    return Nc



def StandWR(M): # Standardized Metrics

    '''
    Standardized version of the original Wright and Reeves metrics, to discount the effect of matrix size.
    This metrics is equal to zero when nestedness does not differ from expectation and is equal to one for any perfectly nested matrix.
    Small negative values are also feasible and indicate a matrix less nested than random expectations.
    
    This function takes a bidimensional matrix (np.array) and returns the normalized Wright-Reeves overlap indicator, computed for rows.
    '''
    
    r,c = M.shape
    k = np.zeros(c)
    N_c_max = 0 # Value that the index would take if the matrix were perfectly nested (initiliazied to zero)
    Nc = WrightReeves(M) # Value of the index for the input matrix
    E_c = 0 # Expected value in a set of randomized matrices (initialized to zero)
    
    for a in range(c):
        k[a] = sum(M[:,a])
        N_c_max = N_c_max + sum(M[:,a])*a
        
    for a in range(c-1):
        for b in np.arange(a+1,c):
            E_c = E_c + k[a]*k[b]
            
    E_c = (1./r)*E_c
    C = (Nc - E_c)/(N_c_max - E_c) # Standardized metrics
    
    return C
    
    
    
def StandWR_inv(M): # Standardized metrics where the computation is carried out for columns instead of rows

    '''
    Complementary version of the standadized Wright and Reeves index, where the computation is carried out for columns instead of rows.
    
    This function takes a bidimensional matrix (np.array) and returns the normalized Wright-Reeves overlap indicator, computed for columns.
    '''
    
    r,c = M.shape
    k = np.zeros(r)
    N_c_max = 0 # Value that the index would take if the matrix were perfectly nested (initiliazied to zero)
    Nc = WrightReeves_inv(M) # Value of the index for the input matrix
    E_c = 0 # Expected value in a set of randomized matrices (initialized to zero)
    
    for a in range(r):
        k[a] = sum(M[a,:])
        N_c_max = N_c_max + sum(M[a,:])*a
        
    for a in range(r-1):
        for b in np.arange(a+1,r):
            E_c = E_c + k[a]*k[b]
            
    E_c = (1./c)*E_c
    C = (Nc - E_c)/(N_c_max - E_c) # Standardized metrics
    
    return C
    
 
# --------------------------------------------------------------------------------
 
 
# The JDM score treats nestedness as a measure of disassortativity between nodes.
# This index allows for the consideration of a nestedness per node.

def JDM(M): # Introduced in [6]

    '''
    Nestedness as a measure of disassortativity between nodes.
    
    This function takes a bidimensional matrix (np.array) and returns the JDM score.
    '''
    
    r,c = M.shape
    s = r+c
    
    Adj = np.zeros((s,s))
    Adj[0:r, r:s]=M
    Adj = Adj+np.transpose(Adj) # Adjacency matrix
    AdjSquare = np.dot(Adj, Adj) # Square adjacency matrix
    
    n = np.zeros((s,s))
    k = sum(Adj)
    
    for i in range(s):
        for j in range(s):
            if (k[i]!=0 and k[j]!=0):
                n[i,j] = AdjSquare[i,j]/(k[i]*k[j])
                
    Eta = sum(sum(n))/(s*s)
    
    return Eta
    
    

def JDM_Norm(M):

    '''
    Normalized version of the JDM index. The original JDM index is divided by the expected value in the configuration model, with the aim of discounting the effect of degree heterogeneity.
    This new normalized index has the advantage that, when close to one, it indicates that the matrix represents an uncorrelated random network.
    
    This function takes a bidimensional matrix (np.array) and returns the normalized JDM score.
    
    '''
    
    r,c = M.shape
    Row_deg = deg.row_deg(M)
    Col_deg = deg.col_deg(M)
    RowDegMean = sum(Row_deg)/r
    ColDegMean = sum(Col_deg)/c
    SquareRowDegMean = sum(Row_deg*Row_deg)/r
    SquareColDegMean = sum(Col_deg*Col_deg)/c
    N_conf = (r*SquareColDegMean + c*SquareRowDegMean)/(RowDegMean*ColDegMean*(r+c)*(r+c)) # Nestedness according to the Configuration model
    
    Eta = JDM(M) # JDM score of the input matrix
    
    Eta_Bip = Eta/N_conf # Normalized nestedness (using the Configuration model)
    
    return Eta_Bip
    
    
# --------------------------------------------------------------------------------

def Tau(M): # Introduced in [7]

    '''
    For a packed matrix, the τ-temperature of the matrix is proportional to the Manhattan distance.
    Various normalizations are possible, which involve the computation of the index for an adequate random matrix.
    
    This function takes a bidimensional matrix (np.array) and returns its τ-temperature.
    '''
    
    r,c = M.shape
    D = np.zeros((r,c))
    
    M_p = M[np.argsort(-M.sum(axis=1))]
    M_o = M_p[:, np.argsort(-M_p.sum(axis=0))] # Ordered matrix

    for i in range(r):
        for j in range(c):
            D[i,j] = M_o[i,j]*(i+j+2) # Manhattan distance (since the indexing start from 0, we add 1 to both i and j)
    Distance = sum(sum(D)) # Sum of the Manhattan distances of all the occupied sites
    
    return Distance
    
    
# --------------------------------------------------------------------------------


# NODF is the most popular nestedness metrics and it is based on the overalp between rows and columns.
# It allows to compute nestedness independently among rows and among columns.

def NODF(M): # Introduced in [8]

    '''
    NODF measures the average percentage of shared contacts between pairs of rows and pairs of columns that have a decreasing degree ordering.
    This score ranges from 0 to 100, where 100 corresponds to a perfectly nested network.
    
    This function takes a bidimensional matrix (np.array) and returns its NODF score.
    '''

    rw,cl=M.shape
    colN = 0
    rowN = 0
    
    Rdeg = deg.row_deg(M)
    Cdeg = deg.col_deg(M)
    
    # Find NODF column score
    for i in range(cl): # At a left position with respect to column j
      	for j in range(cl):
                  if (Cdeg[i]>Cdeg[j])&(Cdeg[j]>0): # DF =! to zero, then NP =! to zero
                      colN = colN + (M[:,i]*M[:,j]).sum()/Cdeg[j]
    
    # NODF_COL = (2*colN/(cl*(cl-1)))*100
    
    # Find NODF row score
    for i in range(rw): # At an upper position with respect to row j
        for j in range(rw):
                if (Rdeg[i]>Rdeg[j])&(Rdeg[j]>0): # DF =! to zero, then NP =! to zero
                    rowN = rowN + (M[i,:]*M[j,:]).sum()/(Rdeg[j])
    
    # NODF_ROW = (2*rowN/(rw*(rw-1)))*100
    
    # Find NODF
    NODF=100*(2*(rowN + colN)/(cl*(cl-1) + rw*(rw-1) ))
    
    return  NODF
    
    
    

def NODF_hetero(M):

    '''
    Complementary measure of the NODF index, where the term heterogeneous is due to the fact that degenerate rows and columns are excluded.
    
    This function takes a bidimensional matrix (np.array) and returns its heterogeneous NODF score
    '''
    
    Rdeg = deg.row_deg(M)
    Cdeg = deg.col_deg(M)
    
    x=np.sum(Rdeg==0) # Number of degenerate rows
    y=np.sum(Cdeg==0) # Number of degenerate columns
    
    rw,cl=M.shape
    colN = 0
    rowN = 0
    
    # Find NODF column score
    for i in range(cl): # At a left position with respect to column j
      	for j in range(cl):
                  if (Cdeg[i]>Cdeg[j])&(Cdeg[j]>0): # DF =! to zero, then NP =! to zero
                      colN = colN + (M[:,i]*M[:,j]).sum()/Cdeg[j]
    
    # NODF_COL = (2*colN/(cl*(cl-1)))*100
    
    # Find NODF row score
    for i in range(rw): # At an upper position with respect to row j
        for j in range(rw):
                if (Rdeg[i]>Rdeg[j])&(Rdeg[j]>0): # DF =! to zero, then NP =! to zero
                    rowN = rowN + (M[i,:]*M[j,:]).sum()/(Rdeg[j])
    
    # NODF_ROW = (2*rowN/(rw*(rw-1)))*100
    
    
    rw=rw-x # Remove degenerate rows from the normalization term
    cl=cl-y # Remove degenerate columns from the normalization term
    
    # Find NODF
    NODF=100*(2*(rowN + colN)/(cl*(cl-1) + rw*(rw-1) ))
    
    return  NODF

    
    
def SNODF(M):

    '''
    S-NODF is a simple variant of NODF consists in including also the contribution of pairs of nodes with the same degree. For pairs of nodes with similar degree the pairwise contributions to S-NODF are more stable with respect to all perturbations of node degree as compared to the pairwise contributions to the original NODF.
    This score ranges from 0 to 100, where 100 corresponds to a perfectly nested network.
    
    This function takes a bidimensional matrix (np.array) and returns its S-NODF score
    '''
    
    rw,cl=M.shape
    colN=0
    rowN=0
    cols_degr = M.sum(axis=0) # Degree of the cols nodes
    rows_degr = M.sum(axis=1) # Degree of the rows nodes
    
    # Find SNODF column score
    for i in range(cl): # At a left position with respect to column j
        for j in range(cl):
            if (i!=j): # To avoid computing the overlap between the same columns
                if (np.sum(M[:,i])>=np.sum(M[:,j]))&(np.sum(M[:,j])>0): # DF =! to zero, then NP =! to zero
                    if (cols_degr[i]==cols_degr[j]): # To avoid counting twice the same overlap
                        colN = colN + (M[:,i]*M[:,j]).sum()/(2*np.sum(M[:,j]))
                    else:
                        colN = colN + (M[:,i]*M[:,j]).sum()/(np.sum(M[:,j]))
                
                
    # Find SNODF row score
    for i in range(rw): # At an upper position with respect to row j
        for j in range(rw):
            if (i!=j): # To avoid computing the overlap between the same rows
                if (np.sum(M[i,:])>=np.sum(M[j,:]))&(np.sum(M[j,:])>0): # DF =! to zero, then NP =! to zero
                    if (rows_degr[i]==rows_degr[j]): # To avoid counting twice the same overlap
                        rowN = rowN + (M[i,:]*M[j,:]).sum()/(2*np.sum(M[j,:]))
                    else:
                        rowN = rowN + (M[i,:]*M[j,:]).sum()/(np.sum(M[j,:]))
                

                
    # Find SNODF
    NODF=100*(2*(rowN+colN)/(cl*(cl-1) + rw*(rw-1) ))
    
    return  NODF



def SNODF_hetero(M):

    '''
    Complementary measure of the SNODF score, where the term heterogeneous is due to the fact that degenerate rows and columns are excluded.
    
    This function takes a bidimensional matrix (np.array) and returns its heterogeneous S-NODF score
    
    '''
    
    Rdeg = deg.row_deg(M)
    Cdeg = deg.col_deg(M)
    
    x = np.sum(Rdeg==0) # Number of degenerate rows
    y = np.sum(Cdeg==0) # Number of degenerate columns
    
    rw,cl = M.shape
    colN = 0
    rowN = 0
    
    # Find SNODF column score
    for i in range(cl): # At a left position with respect to column j
          for j in range(cl):
            if (i!=j):
                if (Cdeg[i]>=Cdeg[j])&(Cdeg[j]>0): # DF =! to zero, then NP =! to zero
                    if (Cdeg[i]==Cdeg[j]):
                        colN = colN + (M[:,i]*M[:,j]).sum()/(2*Cdeg[j])
                    else:
                        colN = colN + (M[:,i]*M[:,j]).sum()/Cdeg[j]
    
    # Find SNODF row score
    for i in range(rw): # At an upper position with respect to row j
        for j in range(rw):
            if (i!=j):
                if (Rdeg[i]>=Rdeg[j])&(Rdeg[j]>0): # DF =! to zero, then NP =! to zero
                    if (Rdeg[i]==Rdeg[j]):
                        rowN = rowN + (M[i,:]*M[j,:]).sum()/(2*Rdeg[j])
                    else:
                        rowN = rowN + (M[i,:]*M[j,:]).sum()/Rdeg[j]
    
    rw = rw-x # Remove degenerate rows from the normalization term
    cl = cl-y # Remove degenerate columns from the normalization term
    
    # Find SNODF
    SNODF = 100*(2*(rowN + colN)/(cl*(cl-1) + rw*(rw-1) ))
    
    return  SNODF



def Glob_N(M): # Introduced in [9]

    '''
    The Global Nestedness Fitness is a variant of NODF that compares the observed level of nestedness with the expected value under a suitable null model.
    This new metrics weighs linearly the contribution of rows and columns (instead of quadratic as in NODF).
    
    This function takes a bidimensional matrix (np.array) and returns its global nestedness score.
    '''
    
    rw,cl=M.shape
    colN=0
    rowN=0
    cols_degr = M.sum(axis=0) # Degree of the cols nodes
    rows_degr = M.sum(axis=1) # Degree of the rows nodes
    
    # Find N col score
    for i in range(cl): # At a left position with respect to column j
        for j in range(cl):
            if (cols_degr[i]>cols_degr[j]) & (cols_degr[j]>0): # Heaviside
                colN = colN + (np.sum((M[:,i]*M[:,j]),dtype=float)-((cols_degr[i]*cols_degr[j])/rw))/cols_degr[j] # Paired overlap
    
    N_COL = colN/(cl-1)
    
    
    # Find N row score
    for i in range(rw): # At an upper position with respect to row j
        for j in range(rw):
            if (rows_degr[i]>rows_degr[j]) & (rows_degr[j]>0): # Heaviside
                rowN = rowN + (np.sum((M[i,:]*M[j,:]),dtype=float)-((rows_degr[i]*rows_degr[j])/cl))/rows_degr[j] # Paired overlap
    
    N_ROW = rowN/(rw-1)
    
    # Find N
    N=(N_COL+N_ROW)*(2./(rw+cl))
    
    return N



def Glob_N_hetero(M):

    '''
    Complementary measure of the global nestedness fitness, where the term heterogeneous is due to the fact that degenerate rows and columns are excluded.
    
    This function takes a bidimensional matrix (np.array) and returns its heterogeneous global nestedness score.
    '''
    
    rw,cl = M.shape
    colN = 0
    rowN = 0
    cols_degr = M.sum(axis=0) # Degree of the cols nodes
    rows_degr = M.sum(axis=1) # Degree of the rows nodes
    
    x = np.sum(rows_degr==0) # Number of degenerate rows
    y = np.sum(cols_degr==0) # Number of degenerate columns
    
    # Find N col score
    for i in range(cl): # At a left position with respect to column j
        for j in range(cl):
            if (cols_degr[i]>cols_degr[j]) & (cols_degr[j]>0): # Heaviside
                colN = colN + (np.sum((M[:,i]*M[:,j]),dtype=float)-((cols_degr[i]*cols_degr[j])/rw))/cols_degr[j] # Paired overlap
                
    # Find N row score
    for i in range(rw): # At an upper position with respect to row j
        for j in range(rw):
            if (rows_degr[i]>rows_degr[j]) & (rows_degr[j]>0): # Heaviside
                rowN = rowN + (np.sum((M[i,:]*M[j,:]),dtype=float)-((rows_degr[i]*rows_degr[j])/cl))/rows_degr[j] # Paired overlap
                
                
    cl = cl-y # Remove degenerate columns from the normalization term
    rw = rw-x # Remove degenerate rows from the normalization term
    
    N_COL = colN/(cl-1)
    N_ROW = rowN/(rw-1)
    
    # Find N
    N = (N_COL+N_ROW)*(2./(rw+cl))
    
    return N


# --------------------------------------------------------------------------------


'''
References:

[1] Lomolino, Mark V. "Investigating causality of nestedness of insular communities: selective immigrations or extinctions?." Journal of biogeography 23.5 (1996): 699-703.
[2] Brualdi, Richard A., and James G. Sanderson. "Nested species subsets, gaps, and discrepancy." Oecologia 119.2 (1999): 256-264.
[3] Staniczenko, Phillip PA, Jason C. Kopp, and Stefano Allesina. "The ghost of nestedness in ecological networks." Nature communications 4.1 (2013): 1-6.
[4] Patterson, Bruce D., and Wirt Atmar. "Nested subsets and the structure of insular mammalian faunas and archipelagos." Biological journal of the Linnean society 28.1-2 (1986): 65-82.
[5] Wright, David H., and Jaxk H. Reeves. "On the meaning and measurement of nestedness of species assemblages." Oecologia 92.3 (1992): 416-428.
[6] Jonhson, Samuel, Virginia Domínguez-García, and Miguel A. Muñoz. "Factors determining nestedness in complex networks." PloS one 8.9 (2013): e74025.
[7] Corso, Gilberto, and Nicholas F. Britton. "Nestedness and τ-temperature in ecological networks." Ecological complexity 11 (2012): 137-143.
[8] Almeida‐Neto, Mário, et al. "A consistent metric for nestedness analysis in ecological systems: reconciling concept and measurement." Oikos 117.8 (2008): 1227-1239.
[9] Solé-Ribalta, Albert, et al. "Revealing in-block nestedness: detection and benchmarking." Physical review E 97.6 (2018): 062302.



For a comprehensive review of nestedness and all of its metrics, see:

- Mariani, Manuel Sebastian, et al. "Nestedness in complex networks: observation, emergence, and implications." Physics Reports 813 (2019): 1-90.
- Ulrich, Werner, Mário Almeida‐Neto, and Nicholas J. Gotelli. "A consumer's guide to nestedness analysis." Oikos 118.1 (2009): 3-17.
- Payrató‐Borràs, Clàudia, Laura Hernández, and Yamir Moreno. "Measuring nestedness: A comparative study of the performance of different metrics." Ecology and evolution 10.21 (2020): 11906-11921.
'''
