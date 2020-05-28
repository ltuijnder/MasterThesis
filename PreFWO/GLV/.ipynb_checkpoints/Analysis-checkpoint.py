
# Imperial code for the analysis of GLV data.


# Point 1 Compute F and Y given the time series
def F(timeseries):
    # Timeseries is  expected to be a numpy with where rows are the observations, First column is the timeseries points and subsequent columns the abundancy of a species.
    
    ts = timeseries[:,1:].T # Remove the first column since it is the time, transpose so that time is now the column
    dt = np.diff(timeseries[:,0])# compute the difference in time points
    # Compute the log difference and return the change per time. 
    dln = np.diff(np.log(ts))
    return dln/dt


def Y_second(timeseries):
    # When computing Y we want to know what kind of parameters we want to infer.
    # This funciton infers the parameters of the a second order descritesed Generalised Lotka Voltera model
    
    # For second order data matrix Y, we need three parts: The growth rate, just the timeseries and the squared timeseries
    ts = timeseries[:,1:].T # acquire timeseries
    ts2 = np.power(ts,2)
    grow = np.ones([1, ts.shape[1]])
    
    # Append them
    Y_all = np.append(grow,ts,axis=0)
    Y_all = np.append(Y_all,ts2, axis=0)
    return np.delete(Y_all, -1, axis = 1) # return with the last element removed. Else dimension do not match since F takes a difference

F_ts=F(ts)
print(F_ts.shape)
Y_ts=Y_second(ts)
print(Y_ts.shape)

# Point 2 Compute the least square estimate of the parameter matrix theta.
def LS_estimate(F,Y):
    YYT =     np.dot(Y,Y.T)
    inv_YYT = np.linalg.inv(YYT)
    FYT =     np.dot(F,Y.T)
    return np.dot(FYT,inv_YYT)

theta_est=LS_estimate(F_ts,Y_ts)
print(theta_est.shape)

# Point 3 Compute F based on our estimate of theta_est
def F_est(theta_est,Y):
    return np.dot(theta_est,Y)

F_ts_est = F_est(theta_est,Y_ts)
print(F_ts_est.shape)

# Point 4 Compute the estimated sigma for each row. 
def Sigma_est(F,F_est,len_theta_row):
    n_obser = F.shape[1]
    norm_factor = np.power(float(n_obser-len_theta_row),-1)
    diff_square = np.power(F-F_est,2)
    return norm_factor*np.sum(diff_square,axis=1) # axis 1 is the row 

sigma_est = Sigma_est(F_ts,F_ts_est,theta_est.shape[1])
print(sigma_est.shape)

# Point 5, Compute estimated variance/ covariance for each row of theta
def Covariance_Tensor_est(Y,sigma_est):
    # Computes the covariance matrix for each row off theta. Store this in a three dimensional tensor.
    # First index is the index for the covariance matrix. Indexes after that are that of the matrix itself.
    
    len_Covar = Y.shape[0]
    inv_YYT= np.linalg.inv(np.dot(Y,Y.T))
    
    for i, sigma_row in enumerate(sigma_est):
        Covar = inv_YYT*sigma_row
        Covar = Covar.reshape((1,len_Covar,len_Covar)) # Make it third order tensor. In order to append it to the All_Covar tensor
        All_Covar = np.append(All_Covar,Covar,axis=0) if i!=0 else Covar # In the first loop All_Covar is the third tensor Covar itself. 
    
    return All_Covar

Cov_Tensor_est = Covariance_Tensor_est(Y_ts,sigma_est)
print(Cov_Tensor_est.shape)


# Point 6 compute the matrix Z for a given hypthesis. 
def Z_hypo(theta_hypo, theta_est, Cov_Tensor_est):
    diff_theta = theta_est-theta_hypo
    var_theta = Cov_Tensor_est.diagonal(offset=0,axis1=1,axis2=2)# We need to take the diagonal of each matrix, which were the indexes on axis 1&2
    return diff_theta/np.sqrt(var_theta)

Z_nullhypo = Z_hypo(np.zeros(shape=theta_est.shape) ,theta_est, Cov_Tensor_est)
print(f"Z_nullhypo.shape={Z_nullhypo.shape}")

def Theta_real(data): # Where data is the dictionary that was returned when running the timeseries generation function
    # Extract the correct matrixes
    r = data['growthrate'] # shape (n,1)
    a = data['allee_factor'] # shape (n,1)
    d = data['immigration_rate'] # shape (n,1)
    B = data['Interaction_Matrix'] # shape (n,n)
    C = data['SecondOrder_matrix'] # shape (n,n)
    
    # define mu, M and N
    mu = r*a-d
    M = B*r.reshape([1,r.shape[0]])
    N = C*r.reshape([1,r.shape[0]])
    # append them and return
    Theta = np.append(mu, M, axis=1)
    Theta = np.append(Theta, N, axis=1)
    return Theta

theta_real = Theta_real(Timeseries)
Z_realhypo = Z_hypo(theta_real, theta_est, Cov_Tensor_est)
print(f"Z_realhypo.shape={Z_realhypo.shape}")

# Point 7, Compute the p-value of the Z-matrix

def P_value(Z_matrix, t_student_df):
    p = 2*(1 - stats.t.cdf(np.abs(Z_matrix),df=t_student_df))# make sure to "from scipy import stats"
    return p

#def cP_value(Z_matrix, t_student_df):
#    p = stats.t.cdf(Z_matrix,df=t_student_df)# make sure to "from scipy import stats"
#    return p

def DF(Y): # returns the degree of freedom used for the t_student distribution
    theta_row, n_obser = Y.shape
    return n_obser-theta_row

p_nullhypo = P_value(Z_nullhypo,DF(Y_ts))
print(p_nullhypo.shape)
p_realhypo = P_value(Z_realhypo,DF(Y_ts))
print(p_realhypo.shape)
#cp_nullhypo = cP_value(Z_nullhypo,DF(Y_ts))
#print(cp_nullhype.shape)
#cp_realhypo = cP_value(Z_realhypo,DF(Y_ts))
#print(cp_realhypo.shape)