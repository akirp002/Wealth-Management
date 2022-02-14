#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cupy as cp
import numpy as np
from cupy import random
import scipy as sc
from scipy import linalg
import matplotlib as plt
import numpy as np
from numpy import genfromtxt
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import invgamma
import matplotlib.pyplot as plt
from sympy import Matrix
from datetime import datetime
import seaborn as sns
import math
import time
from scipy.optimize import minimize, rosen, rosen_der
ordqz = sc.linalg.ordqz
svd = sc.linalg.svd
import pandas as pd
resh  = cp.reshape
norm = cp.random.standard_normal
zeros = cp.zeros
import scipy as sc
inv =  sc.linalg.inv
resh = np.reshape
m = np.matmul
eig = sc.linalg.eig
det = cp.linalg.det
num = cp.asnumpy

import seaborn as sns;import math;import pandas as pd;import scipy as sc;from sklearn.mixture import GaussianMixture;from sklearn.mixture import BayesianGaussianMixture;import time;
import numpy as np;from sklearn.mixture import GaussianMixture;from sklearn import mixture
#m= np.matmul;resh = np.reshape;norm= np.random.standard_normal;inv  = np.linalg.inv ;det = np.linalg.det
import time;import statsmodels.api as sm;from statsmodels.tsa.api import VAR
#arr = np.array;zeros = np.zeros


def ab(a,b,loops):
    for i in range(loops):
        c = cp.matmul(a,b,out=None)
    return c
def set_seed():
    t = 1000 * time.time() # current time in milliseconds
    y = int(t) % 2**32
    cp.random.seed(y) 


# In[3]:


def plot(z1,z2):
    fig,ax = plt.subplots(10,2,figsize=(15,15))

    idx = 0
    ax[0,0].plot(cp.asnumpy(R_t[z1:z2,idx]))
    ax[0,0].plot(cp.asnumpy(y_pred_1[z1:z2,idx]))
    ax[0,0].title.set_text('India')
    idx = 1
    ax[1,0].plot(cp.asnumpy(R_t[z1:z2,idx]))
    ax[1,0].plot(cp.asnumpy(y_pred_1[z1:z2,idx]))
    ax[1,0].title.set_text('Taiwan')
    idx = 2
    ax[2,0].plot(cp.asnumpy(R_t[z1:z2,idx]))
    ax[2,0].plot(cp.asnumpy(y_pred_1[z1:z2,idx]))
    ax[2,0].title.set_text('Indonesia')
    idx = 4
    ax[3,0].plot(cp.asnumpy(R_t[z1:z2,idx]))
    ax[3,0].plot(cp.asnumpy(y_pred_1[z1:z2,idx]))
    ax[3,0].title.set_text('China')
    idx = 6
    ax[4,0].plot(cp.asnumpy(R_t[z1:z2,idx]))
    ax[4,0].plot(cp.asnumpy(y_pred_1[z1:z2,idx]))
    ax[4,0].title.set_text('Thailand') 
    idx = 8
    ax[5,0].plot(cp.asnumpy(R_t[z1:z2,idx]))
    ax[5,0].plot(cp.asnumpy(y_pred_1[z1:z2,idx]))
    ax[5,0].title.set_text('France')
    idx = 10
    ax[6,0].plot(cp.asnumpy(R_t[z1:z2,idx]))
    ax[6,0].plot(cp.asnumpy(y_pred_1[z1:z2,idx]))
    ax[6,0].title.set_text('Canada')
    idx = 12
    ax[7,0].plot(cp.asnumpy(R_t[z1:z2,idx]))
    ax[7,0].plot(cp.asnumpy(y_pred_1[z1:z2,idx]))
    ax[7,0].title.set_text('Korea')
    idx = 14
    ax[8,0].plot(cp.asnumpy(R_t[z1:z2,idx]))
    ax[8,0].plot(cp.asnumpy(y_pred_1[z1:z2,idx]))
    ax[8,0].title.set_text('Australia')
    idx = 16
    ax[9,0].plot(cp.asnumpy(R_t[z1:z2,idx]))
    ax[9,0].plot(cp.asnumpy(y_pred_1[z1:z2,idx]))
    ax[9,0].title.set_text('Japan')
    idx = 17
    ax[0,1].plot(cp.asnumpy(R_t[z1:z2,idx]))
    ax[0,1].plot(cp.asnumpy(y_pred_1[z1:z2,idx]))
    ax[0,1].title.set_text('U.K.')
    idx = 18
    ax[1,1].plot(cp.asnumpy(R_t[z1:z2,idx]))
    ax[1,1].plot(cp.asnumpy(y_pred_1[z1:z2,idx]))
    ax[1,1].title.set_text('Germany')
    idx = 20
    ax[2,1].plot(cp.asnumpy(R_t[z1:z2,idx]))
    ax[2,1].plot(cp.asnumpy(y_pred_1[z1:z2,idx]))
    ax[2,1].title.set_text('Eurozone')
    idx = 26
    ax[3,1].plot(cp.asnumpy(R_t[z1:z2,idx]))
    ax[3,1].plot(cp.asnumpy(y_pred_1[z1:z2,idx]))
    ax[3,1].title.set_text('SPY')
    idx = 27
    ax[4,1].plot(cp.asnumpy(R_t[z1:z2,idx]))
    ax[4,1].plot(cp.asnumpy(y_pred_1[z1:z2,idx]))
    ax[4,1].title.set_text('W500')

    #plots(idx)


# In[4]:


def OLS_scale(Z0,Z1):
    nump = cp.asnumpy
    Z0 = nump(Z0)
    Z1 = nump(Z1)
    N = Z0.shape[1]
    #T= Z0.shape[0]-100
    if np.any(np.isnan(Z1)) == True:
        nan_idx =(np.where(np.isnan(Z1))[0][-1])+1
        Z1 = Z1[nan_idx:]
        Z0 = Z0[nan_idx:]
    X = cp.asnumpy(  abs(  resh(Z0[:],[-1,1]) )  )
    Y  =cp.asnumpy(  abs(resh( ( (resh(Z1[:],[-1,1]))    ),[-1,1])  ) )
    c = m( m(inv(m(  X.T   ,   X)),X.T), Y )
    
    return c

    


# In[5]:


def sort_regressor(list_R,m,F):
    
    r_i = int(list_R[0])
    try:
        r_j = int(list_R[1])
    except:
        n1  = np.arange(900)[r_i]  # Row Coordinate -> Return Index
        n2 = np.arange(900)[m] # Column Coordinate -> Macro Index
        col_idx = tuple(resh((   resh(X_t.shape[1]*n1,[-1])   +resh(n2,[-1,1]) ).T,[-1]) )
        o = int(len(col_idx)/len(tuple(n1)))
        row_idx = np.tile(n1,o)
        F[row_idx,col_idx] = 1
        return F
    n1  = np.arange(900)[r_i:r_j]  # Row Coordinate -> Return Index
    n2 = np.arange(900)[m] # Column Coordinate -> Macro Index
    col_idx = tuple(resh((   resh(X_t.shape[1]*n1,[-1])   +resh(n2,[-1,1]) ).T,[-1]) )
    o = int(len(col_idx)/len(tuple(n1)))
    row_idx = np.tile(n1,o)
    #print(row_idx)
    #print(col_idx)
    F[row_idx,col_idx] = 1
    return F


# In[6]:


def prop(para,BB):
    F = arrange_F()
    #def gen_dist(para,F):
    #P = 4000
    # Parameter Block
    J = 15
    port_idx = tuple(np.array([0,1,2,4,6,8,10,12,14,16,17,18,20]))
    a_1 = n;
    a_2 = a_1+1
    a_3 = a_2+1
    a_4 = a_3+n
    a_5 = a_4+J*n
    a_6 = a_5+1
    a_7 = a_6+n
    a_8 = a_7+1
    rho_a0 = resh(cp.diagflat(  resh(para[0:a_1],[n,1])    ),[1,n,n]) #0:a1
    rho_a = (resh(para[a_1:a_2],[1,1]) )  #a1:a2
    m_alpha = resh(para[a_2:a_3],[1]) #a2:a3
    m_0 = resh(para[a_3:a_4],[1,n,1]) #a3:a4
    mu = resh(para[a_4:a_5],[J,n,1]) #a4:a5
    SIG_a = resh(para[a_5:a_6],[1])*resh(cp.eye(k_bar),[1,k_bar,k_bar]) #a5:a6
    SIG_a0 = resh(cp.diagflat(resh(para[a_6:a_7],[n,1])),[1,n,n])  #a6:a7
    c = para[a_7:a_8] #a7:a8
    c_m = 1e0;
    #Initialize
    #V_m = c_m*resh(cp.diagflat(1*abs(norm([n,1]))),[1,n,n])
    c_lam = 1e0
    V_m = c_m*cp.eye(n)
    alpha = 1e-2*norm([J,k_bar,1])
    alpha_0 = 1e-2*norm([J,n,1])
    Lambda = c_lam*cp.ones([J,n,1])
    Lambda_lag = Lambda+0
    cp.random.seed(1)
    f_lag = 1e-1*norm([J,n,1])
    P0_t_a = 1e-1*norm([J,k_bar,k_bar])
    P0_t_a0= 1e-1*norm([J,n,n])
    T  =293
    scaled = cp.ones([J,1,1])
    y_pred = cp.zeros([T,n,1])
    SSE = cp.zeros([T,1])
    for i in range(J):
        scaled[i]*=(c**i)
    SIG_f = scaled*ab(BB,cp.swapaxes(BB,2,1),1)
    X_bar = cp.kron(cp.eye(n),X_t[0,:])
    start_time = time.time()
    for t in range(293):
        if t==0:
            X_bar = cp.kron(cp.eye(n),X_t[0,:])
            X_bar = X_bar*F
            X_bar = resh(X_bar,[1,n,k_bar])

        if t%4==0:
            X_bar = cp.kron(cp.eye(n),(X_t[t,:]))
            X_bar = X_bar*F
            X_bar = resh(X_bar,[1,n,k_bar])

        ### Kalman Filter ###
        alpha = m_alpha+rho_a*alpha
        alpha_0 = m_0+ab(rho_a0,alpha_0,1)
        f=  alpha_0+ab(X_bar,alpha,1)+mu
        FF = (f*cp.eye(n))
        FF_lag = (f_lag*cp.eye(n))
        y = ab(FF,Lambda,1)-ab(FF_lag,Lambda,1) 
        y[cp.where(y<-1)[0],cp.where(y<-1)[1]] = -1 
        y[cp.where(y>.5)[0],cp.where(y>.5)[1]] = .4        
        
        f=  ab(X_bar,alpha,1)+mu

        if t==0:
            w = (1/J)*cp.ones([J])
        y_pred[t,:] = cp.sum((resh(w,[J,1,1])*y),0)

        #Compute Errors
        idkkk = cp.where(cp.isnan(resh(R_t[t,:],[1,n,1])))
        errs = (y-resh(R_t[t,:],[1,n,1]))
        errs[:,idkkk[1],idkkk[2]]  = 0
        ### Updates ###
        P_t_a = (rho_a**2)*P0_t_a+SIG_a
        P_t_a0 = ab(ab(rho_a0,P0_t_a0,1),cp.swapaxes(rho_a0,2,1),1)+SIG_a0
        ### f block ####
        P_t_f = ab(ab(X_bar,P_t_a,1),cp.swapaxes(X_bar,2,1),1)+SIG_f+P_t_a0
        L_t_f = ab((Lambda*cp.eye(n)),P_t_f,1)
        D_t_f = ab(ab((Lambda*cp.eye(n)),P_t_f*cp.eye(n),1),
                   (Lambda*cp.eye(n)),1)+resh(V_m,[1,n,n])
        ### alpha0 block ###
        #P_t_a0 = ab(ab(rho_a0,P0_t_a0,1),cp.swapaxes(rho_a0,2,1),1)+SIG_a0
        L_t_a0 = P_t_a0
        D_t_a0 = ( ab(ab((Lambda*cp.eye(n)),P_t_f*cp.eye(n),1),(Lambda*cp.eye(n)),1)
                +resh(V_m,[1,n,n]) )
        ### alpha block ###
        #P_t_a = (rho_a**2)*P0_t_a+SIG_a
        L_t_a = ab((Lambda*cp.eye(n)),ab(X_bar,P_t_a,1),1)
        D_t_a = D_t_a0
        '''
        RR = np.random.choice(np.arange(0, J), p=resh(cp.asnumpy(w),[-1]),size=P,replace = True)
        eps_f = np.zeros([P,n]);nn_lag=0
        if t==0:
            f_dist_lag = norm([P,n,1])
        for i in range(J):
            nn = np.where(RR==i)[0].shape[0]
            if i==0:
                idxx = nn
            else:
                idxx += nn
            eps_f[(idxx-nn):(idxx),]  =  resh(np.random.multivariate_normal(
                                     cp.asnumpy(resh(mu[i],[-1])),
                                     cp.asnumpy( SIG_f[i] ) ,
                                    #cp.asnumpy( P_t_f[i] ),
                                     nn 
                                     ) ,[-1,n])
        f_dist =  resh(cp.sum(resh(w,[J,1,1])*f,0),[1,n,1])+(c_lam**0)*cp.array(np.reshape(eps_f,[P,n,1]))
        #f_dist = (cp.sum(resh(w,[J,1,1])*alpha_0,0)+ab(X_bar,cp.sum(resh(w,[J,1,1])*alpha,0),1)+epsilon_f )
        y_dist = c_lam*f_dist -c_lam*f_dist_lag
        y_proj[t,:] = y_dist
        f_dist_lag =  f_dist+0
        '''
        ### Updates ### 
        alpha_0 = alpha_0 + ab(ab(cp.swapaxes(L_t_a0,2,1),cp.linalg.inv(D_t_a0),1),errs  ,1)
        alpha = alpha + ab(ab(cp.swapaxes(L_t_a,2,1),cp.linalg.inv(D_t_a),1),errs  ,1)
        f=  f + ab(ab(cp.swapaxes(L_t_f,2,1),cp.linalg.inv(D_t_f),1),errs  ,1)
        f_lag = f+0 
        #Lambda_lag = Lambda+0
        P0_t_f = P_t_f - ab(ab(cp.swapaxes(L_t_f,2,1),cp.linalg.inv(D_t_f),1),L_t_f,1)
        P0_t_a = P_t_a - ab(ab(cp.swapaxes(L_t_a,2,1),cp.linalg.inv(D_t_a),1),L_t_a,1)
        P0_t_a0 = P_t_a0 - ab(ab(cp.swapaxes(L_t_a0,2,1),cp.linalg.inv(D_t_a0),1),L_t_a0,1)


        ### Error Calculation  ####
        SIG_w = c_m*cp.eye(n) +c_lam*SIG_f
        w = cp.exp(-.5*resh((cp.linalg.det(SIG_w)**-.5),[J,1,1])*
           ab(ab(cp.swapaxes(errs,2,1),cp.linalg.inv(SIG_w),1),errs,1) )
        w = w/cp.sum(w)    
    try:
        c = np.zeros([n,1])
        for i in range(n):
            c[i] = OLS_scale(y_pred[50:200,i],R_t[50:200,i])
    except:
        likic = math.nan
        #print('Nope!')  
        return likic,SIG_f
        #return y_pred,likic,SIG_f
        
        
    y_pred_1 = resh(  cp.array(c) ,[-1,n])*resh(y_pred,[-1,n])
    errs = y_pred_1-R_t
    errs[cp.where(cp.isnan(R_t)  )[0],cp.where(cp.isnan(R_t)  )[1]] = 0
    errs[:,list(port_idx)]*=1e1
    errs[cp.where((errs>0  )  )[0],cp.where( (errs>0 )  )[1]] *= 1e0
    
    #likic = cp.sum(-0.5*errs[10:,:]**2)+-.5*np.var(SSE) 

    likic = cp.sum(-5*errs[20:-20,:]**2) +-5*np.var(SSE)

    end_time = time.time()
    #print('likelihood:', likic)
    #print('computing time:', end_time-start_time)
    
    #return y_pred_1,likic,SIG_f
    return likic,SIG_f


# In[7]:


def arrange_F():
    # Macro-Indices
    np.arange(100)[0:3]
    m_aus = 0;    #Australia
    m_cad = 1;    #Canada
    m_china = 2;    #China
    m_germ = 3;   #Germany
    m_euro = 4;   #Euro-Zone
    m_france = 5;   #France
    m_uk = 6;   #U-K
    m_ind = 7;   #India
    m_indo = 8;   #Indonesia
    m_jap = 9;  #Japan
    m_kor = 10;  #Korea
    m_usa = 11;  #U.S.
    m_thai = 12;  #Thailand  
    m_tai = 13;  #Taiwan  
    #m_15 = 44;  #E_L_S, for every-return
    #m_16 = 45;  #Global Fund

    #np.arange(100)[45:51] #Thailand p2
    #np.arange(100)[-1]    #Global Fund
    # Return-Indices
    r_1 = 0     #India...Single
    r_2 = 1     #Taiwan...Single
    r_3 = 4     #Indonesia np.arange(10)[2:r_3]
    r_4 = 6     #China
    r_5 = 8     #Thailand
    r_6 = 10    #France 
    r_7 = 12    #Canada 
    r_8 = 14    #Korea 
    r_9 = 16    #Australia 
    r_10 = 16    #Japan...Single
    r_11 = 17    #UK...Single 
    r_12 = 20    #Germany   np.arange(10)[18:r_12]
    r_13 = 20   #Eurozone...Single   
    r_14 = 21   #MIGAX...Single...USA
    r_15 = 22   #VCSH...Single...USA
    r_16 = 23   #VWOB...Single...Nothing
    r_17 = 24   #BOND...Single...USA
    r_18 = 25   #BND...Single....USA
    r_19 = 26   #SPY...Single....USA
    r_20 = 27   #W500...Single....USA

    # Bonds
    #MIGAX Morgan Stanley Corporate Bond Portfolio          #21                                                   #21
    #VCSH Vanguard Short-Term Corporate Bond Idx Fd ETF     #22
    #VWOB Vanguard Emerging Markets Govt Bd Idx ETF         #23
    #BOND PIMCO Active Bond Exchange-Traded Fund (BOND)     #24
    #BND Vanguard Total Bond Market Index Fund ETF Shares   #25
    #F[r_1,m_ind] = 1              #India
    F = np.zeros([n,k_bar])

    list_R =[r_1,r_1+1] 
    F = sort_regressor(list_R,m_ind,F)
    #F[r_2,m_tai] = 1            #Taiwan
    list_R =[r_2,r_2+1] 
    F = sort_regressor(list_R,m_tai,F)
    #F[2:r_3,m_indo] = 1            #Indonesia
    list_R =[2,r_3] 
    F = sort_regressor(list_R,m_indo,F)
    #F[r_3:r_4,m_china] = 1          #China
    list_R =[r_3,r_4] 
    F = sort_regressor(list_R,m_china,F)

    #F[r_4:r_5,m_thai] = 1        #Thailand
    list_R =[r_4,r_5] 
    F = sort_regressor(list_R,m_thai,F)
    #F[r_5:r_6,m_france] = 1          #France
    list_R =[r_5,r_6] 
    F = sort_regressor(list_R,m_thai,F)
    #F[r_6:r_7,m_cad] = 1          #Canada
    list_R =[r_6,r_7] 
    F = sort_regressor(list_R,m_cad,F)
    #F[r_7:r_8,m_kor] = 1        #Korea
    list_R =[r_7,r_8] 
    F = sort_regressor(list_R,m_kor,F)

    #F[r_8:r_9,m_aus] = 1            #Australia
    list_R =[r_8,r_9] 
    F = sort_regressor(list_R,m_aus,F)
    #F[r_10,m_jap] = 1            #Japan
    list_R =[r_10,r_10+1] 
    F = sort_regressor(list_R,m_jap,F)
    #F[r_11,m_uk] = 1             #UK
    list_R =[r_11,r_11+1] 
    F = sort_regressor(list_R,m_uk,F)
    #F[18:r_12,m_germ] = 1          #Germany
    list_R =[18,r_12] 
    F = sort_regressor(list_R,m_germ,F)
    #F[r_13,m_euro] = 1             #Euro-Zone
    list_R =[r_13,r_13+1] 
    F = sort_regressor(list_R,m_euro,F)
    #F[r_14,m_usa] = 1           #MIGAX
    list_R =[r_14,r_14+1] 
    F = sort_regressor(list_R,m_usa,F)
    #F[r_15,m_usa] = 1           #VCSH
    list_R =[r_15,r_15+1] 
    F = sort_regressor(list_R,m_usa,F)

    #F[r_16,m_usa] = 0           #VWOB...Single...Nothing
    list_R =[r_16,r_16+1] 
    F = sort_regressor(list_R,m_usa,F)
    #F[r_17,m_usa] = 1           #BOND
    list_R =[r_17,r_17+1] 
    F = sort_regressor(list_R,m_usa,F)
    F[r_18,m_usa] = 1           #BND
    list_R =[r_18,r_18+1] 
    F = sort_regressor(list_R,m_usa,F)
    #F[r_19,m_usa] = 1           #SPY
    list_R =[r_19,r_19+1] 
    F = sort_regressor(list_R,m_usa,F)
    #F[r_20,m_usa] = 1           #W500
    list_R =[r_20,r_20+1] 
    F = sort_regressor(list_R,m_usa,F)
    
    
    return cp.array(F)


# In[8]:


#R_t = cp.array(np.load('/home/ajay/Downloads/Threshold Model/R_t.npy'))


# In[ ]:





# In[9]:


#X_t = cp.array(np.load('/home/ajay/Downloads/Threshold Model/X_t.npy'))
X_t = cp.array(np.load('/home/ajay/Downloads/Threshold Model/f2_t.npy') )
Y_t = cp.array(np.load('/home/ajay/Downloads/Threshold Model/Y_t.npy'))
#R_t = cp.array(np.load('/home/ajay/Downloads/Threshold Model/R_t.npy')) *1e2
R_t = cp.array(np.load('/home/ajay/Downloads/Threshold Model/R_t.npy'))
Dates =pd.read_csv('/home/ajay/Downloads/Threshold Model/Dates.csv')

n = R_t.shape[1]
k = X_t.shape[1]
k_bar = n*k


# In[10]:


cp.any(cp.isnan(R_t[:, [0,1,2,3,4,5,6,7
                        ,8,9,10,11
                       ,12,13,14,15,16,17,18,19,20,26]  ]))


# In[11]:


np.any(np.isnan(R_t[:, [0,1,2,3,4,5,6,7,8,9,10]]))


# In[12]:


np.where(np.isnan(R_t[:, [0,1,3,5]])==True)[0]


# In[ ]:





# In[13]:


def initialize_param():
    
    J = 15
    a_1 = n;
    a_2 = a_1+1
    a_3 = a_2+1
    a_4 = a_3+n
    a_5 = a_4+J*n
    a_6 = a_5+1
    a_7 = a_6+n
    a_8 = a_7+1
    npara = a_8
    para = norm([npara,1])
    para[0:a_1] =   1e-1*abs(para[0:a_1]) #rho_a0
    para[a_1:a_2] = 1e-1*abs(para[a_1:a_2]) #rho_a
    summs = -1
    while summs<0:
        para[a_2:a_3] = 1e-3* norm([1]) #m_alpha
        summs = cp.sum(para[a_2:a_3])
    summs = -1
    while summs<0:
        para[a_3:a_4] = 1e-3* norm([n,1]) #m_0
        summs = cp.sum(para[a_3:a_4]) 
    summs = -1
    while summs<0:
        para[a_4:a_5] = 1e0* norm([J*n,1]) #mu
        summs = cp.sum(para[a_4:a_5])
    #(para[a_4:a_5])
    para[a_5:a_6] = 1e0*abs(para[a_5:a_6]) #SIG_a
    para[a_6:a_7] = 1e0*abs(para[a_6:a_7]) #SIG_a0
    para[a_7:a_8] = 1+1e-1*abs(para[a_7:a_8]) #c
    BB = 2e0*norm([1,n,n])    
    return para,BB
    


# In[14]:


para,BB = initialize_param()
npara = para.shape[0]


# In[15]:


#likic,SIG_f = prop(para,BB)
#print(likic)
BB = cp.load('/home/ajay/Downloads/Threshold Model/Estimation Data/BB_sim.npy')[-1]
para = cp.load('/home/ajay/Downloads/Threshold Model/Estimation Data/PostDIST.npy'
)[-1]
BB = resh(BB,[1,n,n])


# In[16]:


P3 = cp.load('/home/ajay/Downloads/Threshold Model/Estimation Data/P3_3.npy')
np.load('/home/ajay/Downloads/Threshold Model/f2_t.npy')


# In[17]:


def props():
    F = arrange_F()
    #def gen_dist(para,F):
    #P = 4000
    # Parameter Block
    J = 15
    port_idx = tuple(np.array([0,1,2,4,6,8,10,12,14,16,17,18,20]))
    a_1 = n;
    a_2 = a_1+1
    a_3 = a_2+1
    a_4 = a_3+n
    a_5 = a_4+J*n
    a_6 = a_5+1
    a_7 = a_6+n
    a_8 = a_7+1
    '''
    rho_a0 = resh(cp.diagflat(  resh(para[0:a_1],[n,1])    ),[1,n,n]) #0:a1
    rho_a = (resh(para[a_1:a_2],[1,1]) )  #a1:a2
    m_alpha = resh(para[a_2:a_3],[1]) #a2:a3
    m_0 = resh(para[a_3:a_4],[1,n,1]) #a3:a4
    mu = resh(para[a_4:a_5],[J,n,1]) #a4:a5
    SIG_a = resh(para[a_5:a_6],[1])*resh(cp.eye(k_bar),[1,k_bar,k_bar]) #a5:a6
    SIG_a0 = resh(cp.diagflat(resh(para[a_6:a_7],[n,1])),[1,n,n])  #a6:a7
    c = para[a_7:a_8] #a7:a8
    '''
    rho_a0 = resh(cp.diagflat(  1e-5*abs(norm([n,1]))    ),[1,n,n]) #0:a1
    rho_a = (resh(1e-5*abs(norm([1]))  ,[1,1]) )  #a1:a2
    m_alpha = resh(0e-1*norm([1]),[1]) #a2:a3
    m_0 = resh(0e-1*abs(norm([n])) ,[1,n,1]) #a3:a4
    mu = resh(1e0*abs(norm([J,n]))   ,[J,n,1]) #a4:a5
    SIG_a = resh(1e0*abs(norm([1])),[1])*resh(cp.eye(k_bar),[1,k_bar,k_bar]) #a5:a6
    SIG_a0 = resh(cp.diagflat(resh(1e0*abs(norm([n])),[n,1])),[1,n,n])  #a6:a7
    c = 1.5 #a7:a8  
    BB = 1e-2*norm([J,n,n])

    c_m = 1e5;
    #Initialize
    #V_m = (c_m**-2)*resh(cp.diagflat(1*abs(norm([n,1]))),[1,n,n])
    V_m = c_m*cp.eye(n)
    c_lam = 1e-1
    alpha = 1e-1*norm([J,k_bar,1])
    alpha_0 = 1e-1*norm([J,n,1])
    Lambda = c_lam*cp.ones([J,n,1])
    Lambda_lag = Lambda+0
    cp.random.seed(1)
    f_lag = 1e0*norm([J,n,1])
    P0_t_a = 1e0*norm([J,k_bar,k_bar])
    P0_t_a0= 1e0*norm([J,n,n])
    T  =296
    scaled = cp.ones([J,1,1])
    y_pred = cp.zeros([T,n,1])
    SSE = cp.zeros([T,1])
    c_p = 1e3
    for i in range(J):
        scaled[i]*=(c**i)
        
    #scaled = norm([J,1,1])
    SIG_f = scaled*ab(BB,cp.swapaxes(BB,2,1),1)
    X_bar = cp.kron(cp.eye(n),X_t[0,:])
    start_time = time.time()
    for t in range(T):
        if t==0:
            X_bar = cp.kron(cp.eye(n),X_t[0,:])
            X_bar = X_bar*F
            X_bar = resh(X_bar,[1,n,k_bar])
        
        if t%200==0:
            X_bar = cp.kron(cp.eye(n),(X_t[t,:]))
            X_bar = X_bar*F
            X_bar = resh(X_bar,[1,n,k_bar])
        ### Kalman Filter ###
        alpha = m_alpha+rho_a*alpha
        alpha_0 = m_0+ab(rho_a0,alpha_0,1)
        f=  alpha_0+ab(X_bar,alpha,1)+mu
        FF = (f*cp.eye(n))
        FF_lag = (f_lag*cp.eye(n))
        y = ab(FF,Lambda,1)-ab(FF_lag,Lambda,1) 
        y[cp.where(y<-1e2)[0],cp.where(y<-1e2)[1]] = -1e2 
        y[cp.where(y>2e1)[0],cp.where(y>2e1)[1]] = 2e1        
        
        f=  ab(X_bar,alpha,1)+mu

        if t==0:
            w = (1/J)*cp.ones([J])
        y_pred[t,:] = cp.sum((resh(w,[J,1,1])*y),0)

        #Compute Errors
        idkkk = cp.where(cp.isnan(resh(R_t[t,:],[1,n,1])))
        errs = (y-resh(R_t[t,:],[1,n,1]))
        errs[:,idkkk[1],idkkk[2]]  = 0
        ### Updates ###
        P_t_a = (rho_a**2)*P0_t_a+SIG_a
        P_t_a0 = ab(ab(rho_a0,P0_t_a0,1),cp.swapaxes(rho_a0,2,1),1)+SIG_a0
        ### f block ####
        P_t_f = ab(ab(X_bar,P_t_a,1),cp.swapaxes(X_bar,2,1),1)+SIG_f+P_t_a0
        L_t_f = ab((Lambda*cp.eye(n)),P_t_f,1)
        D_t_f = ab(ab((Lambda*cp.eye(n)),P_t_f*cp.eye(n),1),
                   (Lambda*cp.eye(n)),1)+resh(V_m,[1,n,n])
        ### alpha0 block ###
        #P_t_a0 = ab(ab(rho_a0,P0_t_a0,1),cp.swapaxes(rho_a0,2,1),1)+SIG_a0
        L_t_a0 = P_t_a0
        D_t_a0 = ( ab(ab((Lambda*cp.eye(n)),P_t_f*cp.eye(n),1),(Lambda*cp.eye(n)),1)
                +resh(V_m,[1,n,n]) )
        ### alpha block ###
        #P_t_a = (rho_a**2)*P0_t_a+SIG_a
        L_t_a = ab((Lambda*cp.eye(n)),ab(X_bar,P_t_a,1),1)
        D_t_a = D_t_a0
        ### Updates ### 
        errs[:,list(port_idx)]*=c_p
        alpha_0 = alpha_0 + ab(ab(cp.swapaxes(L_t_a0,2,1),cp.linalg.inv(D_t_a0),1),errs  ,1)
        alpha = alpha + ab(ab(cp.swapaxes(L_t_a,2,1),cp.linalg.inv(D_t_a),1),errs  ,1)
        f=  f + ab(ab(cp.swapaxes(L_t_f,2,1),cp.linalg.inv(D_t_f),1),errs  ,1)
        f_lag = f+0 
        #Lambda_lag = Lambda+0
        
        P0_t_f = P_t_f - ab(ab(cp.swapaxes(L_t_f,2,1),cp.linalg.inv(D_t_f),1),L_t_f,1)
        P0_t_a = P_t_a - ab(ab(cp.swapaxes(L_t_a,2,1),cp.linalg.inv(D_t_a),1),L_t_a,1)
        P0_t_a0 = P_t_a0 - ab(ab(cp.swapaxes(L_t_a0,2,1),cp.linalg.inv(D_t_a0),1),L_t_a0,1)


        ### Error Calculation  ####
        SIG_w = c_m*cp.eye(n) +c_lam*SIG_f
        SIG_f[list(port_idx),list(port_idx)] = 1e-2*c_lam
        w = cp.exp(-.5*resh((cp.linalg.det(SIG_w)**-.5),[J,1,1])*
           ab(ab(cp.swapaxes(errs,2,1),cp.linalg.inv(SIG_w),1),errs,1) )
        w = w/cp.sum(w)   
        
    c = np.zeros([n,1])
    for i in range(n):
        c[i] = OLS_scale(y_pred[20:200,i],R_t[20:200,i])
    y_pred_1 = resh(  cp.array(c) ,[-1,n])*resh(y_pred,[-1,n])
    
    #y_pred_1 = 1e2*resh(y_pred,[T,n])+0   
    errs = y_pred_1-R_t
    errs[cp.where(cp.isnan(R_t)  )[0],cp.where(cp.isnan(R_t)  )[1]] = 0
    errs[:,list(port_idx)]*= (c_p**-1)
    likic = cp.sum(-5*(c_m**-1)*errs[20:-20,:]**2) #+-.5*np.var(SSE)

    end_time = time.time()
    print('likelihood:', likic)
    print('computing time:', end_time-start_time)
    print('Works:', cp.all(cp.mean(abs(y_pred_1),0)>1))
    return y_pred_1,likic,SIG_f
    #return likic,ab(BB,cp.swapaxes(BB,2,1),1)


# In[18]:


def create_F():
    F = cp.zeros([int(R_t.shape[1]), int(X_t[0,:].shape[0])])    
    m_aus = 0;    #Australia
    m_cad = 1;    #Canada
    m_china = 2;    #China
    m_germ = 3;   #Germany
    m_euro = 4;   #Euro-Zone
    m_france = 5;   #France
    m_uk = 6;   #U-K
    m_ind = 7;   #India
    m_indo = 8;   #Indonesia
    m_jap = 9;  #Japan
    m_kor = 10;  #Korea
    m_usa = 11;  #U.S.
    m_thai = 12;  #Thailand  
    m_tai = 13;  #Taiwan  
    #m_15 = 44;  #E_L_S, for every-return
    #m_16 = 45;  #Global Fund

    #np.arange(100)[45:51] #Thailand p2
    #np.arange(100)[-1]    #Global Fund
    # Return-Indices
    r_1 = 0     #India...Single
    F[r_1,m_ind] =1
    r_2 = 1     #Taiwan...Single
    F[r_2,m_tai] =1
    r_3 = 4     #Indonesia np.arange(10)[2:r_3]
    F[r_3,m_indo] =1
    r_4 = 6     #China
    F[r_4,m_china] = 1
    r_5 = 8     #Thailand
    F[r_5,m_thai] = 1
    r_6 = 10    #France 
    F[r_6,m_france] = 1
    r_7 = 12    #Canada 
    F[r_7,m_cad] = 1
    r_8 = 14    #Korea 
    F[r_8,m_kor] = 1
    r_9 = 16    #Australia 
    F[r_9,m_aus] = 1
    r_10 = 16    #Japan...Single
    F[r_10,m_jap] = 1 
    r_11 = 17    #UK...Single 
    F[r_11,m_uk] = 1
    r_12 = 20    #Germany   np.arange(10)[18:r_12]
    F[r_12,m_germ]  = 1
    r_13 = 20   #Eurozone...Single   
    F[r_13,m_euro] = 1
    r_14 = 21   #MIGAX...Single...USA
    F[r_14,m_usa]  = 1
    r_15 = 22   #VCSH...Single...USA
    F[r_15,m_usa] = 1
    r_16 = 23   #VWOB...Single...Nothing
    F[r_16,m_usa] = 1
    r_17 = 24   #BOND...Single...USA
    F[r_17,m_usa] = 1
    r_18 = 25   #BND...Single....USA
    F[r_18,m_usa] = 1
    r_19 = 26   #SPY...Single....USA
    F[r_19,m_usa] = 1
    r_20 = 27   #W500...Single....USA
    F[r_20,m_usa] = 1
    return F


# In[19]:


F = create_F()
X_bar = resh(X_t[0,:],[-1,1])
ab(F,X_bar,1).shape


# In[ ]:





# In[ ]:





# In[28]:


def props_2(J):
    #F = arrange_F()
    #def gen_dist(para,F):
    #P = 4000
    # Parameter Block
    #J = 12
    port_idx = tuple(np.array([0,1,2,4,6,8,10,12,14,16,17,18,20]))
    a_1 = n;
    a_2 = a_1+1
    a_3 = a_2+1
    a_4 = a_3+n
    a_5 = a_4+J*n
    a_6 = a_5+1
    a_7 = a_6+n
    a_8 = a_7+1
    F = create_F()

    rho_a0 = resh(cp.diagflat(  0e-3*abs(norm([n,1]))    ),[1,n,n]) #0:a1
    rho_a = resh(cp.diagflat(  0e-3*abs(norm([n,1]))    ),[1,n,n]) #a1:a2
    m_alpha = resh(0e-1*norm([1]),[1]) #a2:a3
    m_0 = resh(0e-1*abs(norm([n])) ,[1,n,1]) #a3:a4
    mu = resh(1e0*abs(norm([J,n]))   ,[J,n,1]) #a4:a5
    SIG_a = resh(cp.diagflat(resh(1e0*abs(norm([n])),[n,1])),[1,n,n]) #a5:a6
    SIG_a0 = resh(cp.diagflat(resh(1e0*abs(norm([n])),[n,1])),[1,n,n])  #a6:a7
    c = 1.5 #a7:a8  
    BB = 1e-2*norm([J,n,n])

    c_m = 1e5;
    #Initialize
    #V_m = (c_m**-2)*resh(cp.diagflat(1*abs(norm([n,1]))),[1,n,n])
    V_m = c_m*cp.eye(n)
    c_lam = 1e0
    alpha = 1e0*norm([J,n,1])
    alpha_0 = 1e0*norm([J,n,1])
    Lambda = c_lam*cp.ones([J,n,1])
    Lambda_lag = Lambda+0
    cp.random.seed(1)
    f_lag = 1e0*norm([J,n,1])
    P0_t_a = 1e0*norm([J,n,n])
    P0_t_a0= 1e0*norm([J,n,n])
    P0_t_f= 1e0*norm([J,n,n])
    T  =296
    scaled = cp.ones([J,1,1])
    y_pred = cp.zeros([T,n,1])
    SSE = cp.zeros([T,1])
    c_p = 1e3
    for i in range(J):
        scaled[i]*=(c**i)
        
    #scaled = norm([J,1,1])
    SIG_f = scaled*ab(BB,cp.swapaxes(BB,2,1),1)
    start_time = time.time()
    for t in range(T):
        if t%4==0:
            X_bar = ab(F,resh(X_t[t,:],[-1,1]),1)
            X_bar = resh(X_bar,[1,n,1])
        ### Kalman Filter ###
        alpha = m_alpha+ab(rho_a,alpha,1) 
        alpha_0 = m_0+ab(rho_a0,alpha_0,1)
        f=  alpha_0+alpha*X_bar+mu
        FF = (f*cp.eye(n))
        FF_lag = (f_lag*cp.eye(n))
        y = ab(FF,Lambda,1)-ab(FF_lag,Lambda,1) 
        #y[cp.where(y<-.4)[0],cp.where(y<-.4)[1]] = -.4
        y[cp.where(y>.2)[0],cp.where(y>.2)[1]] = .2        
        
        A = ab(alpha,cp.swapaxes(alpha,2,1),1 )
        if t==0:
            w = (1/J)*cp.ones([J])
        y_pred[t,:] = cp.sum((resh(w,[J,1,1])*y),0)

        #Compute Errors
        idkkk = cp.where(cp.isnan(resh(R_t[t,:],[1,n,1])))
        errs = (y-resh(R_t[t,:],[1,n,1]))
        errs[:,idkkk[1],idkkk[2]]  = 0
        ### Updates ###
        P_t_a = (rho_a**2)*P0_t_a+SIG_a
        P_t_a0 = ab(ab(rho_a0,P0_t_a0,1),cp.swapaxes(rho_a0,2,1),1)+SIG_a0
        ### f block ####
        #P_t_f = ab(ab(X_bar,P_t_a,1),cp.swapaxes(X_bar,2,1),1)+SIG_f+P_t_a0
        X = (resh(X_bar,[1,n,1])*cp.eye(n))
        P_t_f = P_t_a0+ab(X**2,P0_t_a,1)+SIG_f
        
        
        L_t_f = ab((Lambda*cp.eye(n)),P_t_f,1)
        D_t_f = ab(ab((Lambda*cp.eye(n)),P_t_f*cp.eye(n),1),
                   (Lambda*cp.eye(n)),1)+resh(V_m,[1,n,n])
        ### alpha0 block ###
        #P_t_a0 = ab(ab(rho_a0,P0_t_a0,1),cp.swapaxes(rho_a0,2,1),1)+SIG_a0
        L_t_a0 = P_t_a0
        D_t_a0 = ( ab(ab((Lambda*cp.eye(n)),P_t_f*cp.eye(n),1),(Lambda*cp.eye(n)),1)
                +resh(V_m,[1,n,n]) )
        ### alpha block ###
        #P_t_a = (rho_a**2)*P0_t_a+SIG_a
        L_t_a = ab(X,P_t_a,1)
        D_t_a = ab(X**2,P_t_a,1)+resh(V_m,[1,n,n])
        ### Updates ### 
        #errs[:,list(port_idx)]*=c_p
        alpha_0 = alpha_0 + ab(ab(cp.swapaxes(L_t_a0,2,1),cp.linalg.inv(D_t_a0),1),errs  ,1)
        alpha = alpha + ab(ab(cp.swapaxes(L_t_a,2,1),cp.linalg.inv(D_t_a),1),errs  ,1)
        f=  f + ab(ab(cp.swapaxes(L_t_f,2,1),cp.linalg.inv(D_t_f),1),errs  ,1)
        f_lag = f+0 
        #Lambda_lag = Lambda+0
        
        P0_t_f = P_t_f - ab(ab(cp.swapaxes(L_t_f,2,1),cp.linalg.inv(D_t_f),1),L_t_f,1)
        P0_t_a = P_t_a - ab(ab(cp.swapaxes(L_t_a,2,1),cp.linalg.inv(D_t_a),1),L_t_a,1)
        P0_t_a0 = P_t_a0 - ab(ab(cp.swapaxes(L_t_a0,2,1),cp.linalg.inv(D_t_a0),1),L_t_a0,1)


        ### Error Calculation  ####
        SIG_w = c_m*cp.eye(n) +1*c_lam*SIG_f
        #SIG_w = 1e1*cp.eye(n) +0*c_lam*SIG_f
        #SIG_f[list(port_idx),list(port_idx)] = 1e-2*c_lam
        w = cp.exp(-.5*resh((cp.linalg.det(SIG_w)**-.5),[J,1,1])*
           ab(ab(cp.swapaxes(errs,2,1),cp.linalg.inv(SIG_w),1),errs,1) )
        w = w/cp.sum(w)   
        
    c = np.zeros([n,1])
    for i in range(n):
        c[i] = OLS_scale(y_pred[20:200,i],R_t[20:200,i])
    y_pred_1 = resh(  cp.array(c) ,[-1,n])*resh(y_pred,[-1,n])
    #print(c[0:5])
    #y_pred_1 = 1e0*resh(y_pred,[T,n])+0   
    
    errs = y_pred_1-R_t
    errs[cp.where(cp.isnan(R_t)  )[0],cp.where(cp.isnan(R_t)  )[1]] = 0
    errs[:,list(port_idx)]*= (c_p**-1)
    likic = (cp.sum(-.5*1e1*errs[20:200,:]**2)) +-.5*np.var(SSE)

    end_time = time.time()
    print('likelihood:', likic)
    print('computing time:', end_time-start_time)
    print('Works:', cp.all(cp.mean(abs(y_pred_1),0)>.05)   )
    return y_pred_1,likic,SIG_f
    #return likic,ab(BB,cp.swapaxes(BB,2,1),1)


# In[29]:


arr = cp.array;
#y_pred_1,likic,SIG_f= props()
y_pred_1,likic,SIG_f= props_2(4)
plot(200,295)


# In[27]:


y_pred_1,likic,SIG_f= props_2(3)


# In[30]:


def Gen_dist(J):
    # Parameter Block
    port_idx = tuple(np.array([0,1,2,4,6,8,10,12,14,16,17,18,20]))
    a_1 = n;
    a_2 = a_1+1
    a_3 = a_2+1
    a_4 = a_3+n
    a_5 = a_4+J*n
    a_6 = a_5+1
    a_7 = a_6+n
    a_8 = a_7+1
    F = create_F()

    rho_a0 = resh(cp.diagflat(  0e-3*abs(norm([n,1]))    ),[1,n,n]) #0:a1
    rho_a = resh(cp.diagflat(  0e-3*abs(norm([n,1]))    ),[1,n,n]) #a1:a2
    m_alpha = resh(0e-1*norm([1]),[1]) #a2:a3
    m_0 = resh(0e-1*abs(norm([n])) ,[1,n,1]) #a3:a4
    mu = resh(1e0*abs(norm([J,n]))   ,[J,n,1]) #a4:a5
    SIG_a = resh(cp.diagflat(resh(1e0*abs(norm([n])),[n,1])),[1,n,n]) #a5:a6
    SIG_a0 = resh(cp.diagflat(resh(1e0*abs(norm([n])),[n,1])),[1,n,n])  #a6:a7
    c = 1.5 #a7:a8  
    BB = 1e-2*norm([J,n,n])

    c_m = 1e5;
    #Initialize
    #V_m = (c_m**-2)*resh(cp.diagflat(1*abs(norm([n,1]))),[1,n,n])
    V_m = c_m*cp.eye(n)
    c_lam = 1e0
    alpha = 1e-1*norm([J,n,1])
    alpha_0 = 1e-1*norm([J,n,1])
    Lambda = c_lam*cp.ones([J,n,1])
    Lambda_lag = Lambda+0
    cp.random.seed(1)
    f_lag = 1e0*norm([J,n,1])
    P0_t_a = 1e0*norm([J,n,n])
    P0_t_a0= 1e0*norm([J,n,n])
    P0_t_f= 1e0*norm([J,n,n])
    T  =296
    scaled = cp.ones([J,1,1])
    y_pred = cp.zeros([T,n,1])
    SSE = cp.zeros([T,1])
    c_p = 1e3
    P = 5000
    y_dist = np.zeros([P,T,n])
    for i in range(J):
        scaled[i]*=(c**i)
        
    #scaled = norm([J,1,1])
    SIG_f = scaled*ab(BB,cp.swapaxes(BB,2,1),1)
    start_time = time.time()
    for t in range(T):
        if t%1==0:
            X_bar = ab(F,resh(X_t[t,:],[-1,1]),1)
            X_bar = resh(X_bar,[1,n,1])
        ### Kalman Filter ###
        alpha = m_alpha+ab(rho_a,alpha,1) 
        alpha_0 = m_0+ab(rho_a0,alpha_0,1)
        f=  alpha_0+alpha*X_bar+mu
        FF = (f*cp.eye(n))
        FF_lag = (f_lag*cp.eye(n))
        y = ab(FF,Lambda,1)-ab(FF_lag,Lambda,1) 
        y[cp.where(y<-1)[0],cp.where(y<-1)[1]] = -1 
        y[cp.where(y>.3)[0],cp.where(y>.3)[1]] = .3        
        
        A = ab(alpha,cp.swapaxes(alpha,2,1),1 )
        if t==0:
            w = (1/J)*cp.ones([J])
        y_pred[t,:] = cp.sum((resh(w,[J,1,1])*y),0)

        #Compute Errors
        idkkk = cp.where(cp.isnan(resh(R_t[t,:],[1,n,1])))
        errs = (y-resh(R_t[t,:],[1,n,1]))
        errs[:,idkkk[1],idkkk[2]]  = 0
        ### Updates ###
        P_t_a = (rho_a**2)*P0_t_a+SIG_a
        P_t_a0 = ab(ab(rho_a0,P0_t_a0,1),cp.swapaxes(rho_a0,2,1),1)+SIG_a0
        ### f block ####
        #P_t_f = ab(ab(X_bar,P_t_a,1),cp.swapaxes(X_bar,2,1),1)+SIG_f+P_t_a0
        X = (resh(X_bar,[1,n,1])*cp.eye(n))
        P_t_f = P_t_a0+ab(X**2,P0_t_a,1)+SIG_f
        
        
        L_t_f = ab((Lambda*cp.eye(n)),P_t_f,1)
        D_t_f = ab(ab((Lambda*cp.eye(n)),P_t_f*cp.eye(n),1),
                   (Lambda*cp.eye(n)),1)+resh(V_m,[1,n,n])
        ### alpha0 block ###
        #P_t_a0 = ab(ab(rho_a0,P0_t_a0,1),cp.swapaxes(rho_a0,2,1),1)+SIG_a0
        L_t_a0 = P_t_a0
        D_t_a0 = ( ab(ab((Lambda*cp.eye(n)),P_t_f*cp.eye(n),1),(Lambda*cp.eye(n)),1)
                +resh(V_m,[1,n,n]) )
        ### alpha block ###
        #P_t_a = (rho_a**2)*P0_t_a+SIG_a
        L_t_a = ab(X,P_t_a,1)
        D_t_a = ab(X**2,P_t_a,1)+resh(V_m,[1,n,n])
        ### Updates ### 
        #errs[:,list(port_idx)]*=c_p
        alpha_0 = alpha_0 + ab(ab(cp.swapaxes(L_t_a0,2,1),cp.linalg.inv(D_t_a0),1),errs  ,1)
        alpha = alpha + ab(ab(cp.swapaxes(L_t_a,2,1),cp.linalg.inv(D_t_a),1),errs  ,1)
        f=  f + ab(ab(cp.swapaxes(L_t_f,2,1),cp.linalg.inv(D_t_f),1),errs  ,1)
        f_lag = f+0 
        #Lambda_lag = Lambda+0
        
        P0_t_f = P_t_f - ab(ab(cp.swapaxes(L_t_f,2,1),cp.linalg.inv(D_t_f),1),L_t_f,1)
        P0_t_a = P_t_a - ab(ab(cp.swapaxes(L_t_a,2,1),cp.linalg.inv(D_t_a),1),L_t_a,1)
        P0_t_a0 = P_t_a0 - ab(ab(cp.swapaxes(L_t_a0,2,1),cp.linalg.inv(D_t_a0),1),L_t_a0,1)
        
        RR = np.random.choice(np.arange(0, J), p=resh(cp.asnumpy(w),[-1]),size=P,replace = True)
        eps_f = np.zeros([P,n]);nn_lag=0
        alpha_0_f = np.zeros([P,n])
        alpha_f = np.zeros([P,n])
        eps =  np.zeros([P,n])
        for i in range(J):
            nn = np.where(RR==i)[0].shape[0]
            if i==0:
                idxx = nn
                eps_f[:(idxx),] = resh(np.random.multivariate_normal(
                                     cp.asnumpy(resh(mu[i],[-1])),
                                     cp.asnumpy( SIG_f[i] ) ,
                                     #cp.asnumpy( P_t_f[i] ) ,
                                     nn 
                                     ) ,[-1,n])

                alpha_0_f[:(idxx),]  =  resh(np.random.multivariate_normal(
                                     cp.asnumpy(resh(alpha_0[i],[-1])),
                                     cp.asnumpy( P_t_a0[i] ) ,
                                     nn 
                                     ) ,[-1,n])
                alpha_f[:(idxx),]  =  resh(np.random.multivariate_normal(
                                     cp.asnumpy(resh(alpha[i],[-1])),
                                     cp.asnumpy( P_t_a[i] ) ,
                                     nn 
                                     ) ,[-1,n])


            else:
                idxx += nn
                eps_f[(idxx-nn):(idxx),]  =  resh(np.random.multivariate_normal(
                                     cp.asnumpy(resh(mu[i],[-1])),
                                     cp.asnumpy( SIG_f[i] ) ,
                                     #cp.asnumpy( P_t_f[i] ) ,
                                     nn 
                                     ) ,[-1,n])
                alpha_0_f[(idxx-nn):(idxx),]  =  resh(np.random.multivariate_normal(
                                     cp.asnumpy(resh(alpha_0[i],[-1])),
                                     cp.asnumpy( P_t_a0[i] ) ,
                                     nn 
                                     ) ,[-1,n])
                alpha_f[(idxx-nn):(idxx),]  =  resh(np.random.multivariate_normal(
                                     cp.asnumpy(resh(alpha[i],[-1])),
                                     cp.asnumpy( P_t_a[i] ) ,
                                     nn 
                                     ) ,[-1,n])
            ZZ = (alpha_0_f+(alpha_f*cp.asnumpy(resh(X_bar,[1,n])  ))    +eps_f)
            #eps_y = (ZZ-np.mean(ZZ,0))
            eps_y = ZZ
            y_dist[:,t,:] = eps_y


        ### Error Calculation  ####
        SIG_w = c_m*cp.eye(n) +1*c_lam*SIG_f
        #SIG_w = 1e1*cp.eye(n) +0*c_lam*SIG_f
        #SIG_f[list(port_idx),list(port_idx)] = 1e-2*c_lam
        w = cp.exp(-.5*resh((cp.linalg.det(SIG_w)**-.5),[J,1,1])*
           ab(ab(cp.swapaxes(errs,2,1),cp.linalg.inv(SIG_w),1),errs,1) )
        w = w/cp.sum(w)   
        
    c = np.zeros([n,1])
    for i in range(n):
        c[i] = OLS_scale(y_pred[20:200,i],R_t[20:200,i])
    y_pred_1 = resh(  cp.array(c) ,[-1,n])*resh(y_pred,[-1,n])
    #print(c[0:5])
    #y_pred_1 = 1e0*resh(y_pred,[T,n])+0   
    
    errs = y_pred_1-R_t
    errs[cp.where(cp.isnan(R_t)  )[0],cp.where(cp.isnan(R_t)  )[1]] = 0
    errs[:,list(port_idx)]*= (c_p**-1)
    likic  = (cp.sum(-.5*1e1*errs[20:200,:]**2)) +-.5*np.var(SSE)
    
    #y_dist = (y_dist - np.mean(y_dist,0))
    #y_dist = (  np.reshape(cp.asnumpy(c) ,[-1,n])   *y_dist)
    #y_dist=  cp.asnumpy(resh(y_pred_1,[1,T,n]))+y_dist
    #y_dist[np.where(y_dist>.3)[0]] = .3
    #y_dist[np.where(y_dist<-1)[0]] = -1
    end_time = time.time()
    print('likelihood:', likic)
    print('computing time:', end_time-start_time)
    print('Works:', cp.all(cp.mean(abs(y_pred_1),0)>.05)   )
    return y_pred_1,y_dist,likic,SIG_f,c


# In[31]:


y_pred_1,y_dist,likic,SIG_f,c = Gen_dist(5)


# In[ ]:


#sns.kdeplot(cp.asnumpy(DIST[:,70,1]),color ='tab:orange')y_pred_1,y_dist,likic,SIG_f,c


# In[32]:


DIST =  (cp.asnumpy(y_pred_1)+resh(c,[1,1,n])*y_dist)
DIST = (DIST/cp.max(abs(DIST),0))
DIST[np.where(DIST>.3)] = .3
DIST[np.where(DIST<-.4)] = -.4


# In[33]:


idx = 26
fig, ax = plt.subplots(figsize = (10,10))
ax.plot(Dates['XX'].iloc[10:],cp.asnumpy(R_t[10:,idx])*1e2,color = "tab:blue",label="Data")

ax.plot(Dates['XX'].iloc[10:],cp.asnumpy(y_pred_1[10:,idx])*1e2,color = "tab:red",label="Forecast")
for i, t in enumerate(ax.get_xticklabels()):
    if (i % 25) != 0 :
        t.set_visible(False)
#ax.set_facecolor('peachpuff')
ax.grid()
ax.legend()


# In[34]:


sns.kdeplot(cp.asnumpy(DIST[:,130,1]),color = 'tab:blue')
sns.kdeplot(cp.asnumpy(DIST[:,70,1]),color ='tab:orange')


# In[ ]:


##### Portfolio Problem #####


# In[35]:


from cvxpy import *
import cvxpy as cp


# In[36]:


idx_P = [0,1,2,4,6,8,10,12,14,16,17,18,20,26,27]
R_p = R_t[:,idx_P]
R_p = R_p[169:]


# In[37]:


R_p[0,:].shape


# In[38]:


np.where(np.isnan(R_t[169:,idx_P])==True)


# In[39]:


idxx = list(np.arange(28))
idxx[23] = 0
idxx[24] = 0


# In[40]:


np.unique(np.where(np.isnan(R_t[169:,list(np.arange(28))])==True)[1])
np.unique(np.where(np.isnan(R_t[169:,idxx])==True)[1])


# In[41]:


import cvxpy as cp
import numpy as np


# In[ ]:





# In[71]:


def generate_wealth(R,t_0):
    W  = np.zeros([127,1])
    W*=0
    W+=1
    while t_0<127:
        W[t_0,:] = R[t_0]*W[t_0-1]
        t_0+=1
        
        
    return W
    


# In[150]:


def generate_Wealth(pro,R):
    T = R.shape[0]
    W_P = np.ones([len(idx_P),T])
    Wealth = 1e0*np.ones([T,1])
    w0 = (1/len(idx_P))*np.ones([len(idx_P),1])
    n = R.shape[1]
    Returns = np.zeros(T)
    P = 1e2*np.eye(n)
    #Assets =np.zeros([15,127])
    Assets = np.zeros([len(idx_PP),127])
    x = cp.Variable(n)
    for t in range(R_p.shape[0]):
        if pro=='Model':
            y_hat = R[t,:]
            P  =1e0*np.cov(y_dist[:,t,idx_PP],rowvar=False) 
            P = P/np.abs(np.array(np.max(np.linalg.eig(P)[0])))
            
            
            skew = np.mean((y_dist[:,0,idx_PP]-np.mean(y_dist[:,0,idx_PP],0))**3,0)*np.eye(15)
            skew = skew/abs(np.array(np.max(skew)))
            q = np.reshape(num(y_hat),[-1])
            x = cp.Variable(n)
            prob =cp.Problem(cp.Minimize(-1*cp.sum(q.T@x )-1e0*q.T@P@q-
                                         1e0*q@skew@q.T),[
            cp.sum( (x)    ) == 1,
            x>=0,
            x<=.5    
                
            ])
            prob.solve()
            w = x.value
            w_1 = w+0
            w_1 = np.sign(w)*w
            w_1 = w_1/np.sum(w_1)
            #w = np.sign(w)*w_1
            if w is None:
                RR_P = num(1+R_p[t,-2])*Wealth[t-1]
            else:
                RR_P = (1+np.sum(np.reshape(num(R_pp[t,:]),[-1])*w)    )
            Wealth[t] = RR_P*Wealth[t-1]
            Assets[:,t] = w+0
            Returns[t] = (1+np.sum(np.reshape(num(R_pp[t,:]),[-1])*w)    )
        if pro == 'Equal-Wieght':
                Wealth[t] =  num(1+np.mean(num(R_p[t,:]))    )*Wealth[t-1]
                Assets[:,t] += 1
                Returns[t] = num(1+np.mean(num(R_p[t,:]))    )
        if pro == 'SPY':
            Wealth[t] =  num(1+R_p[t,-2])*Wealth[t-1]
            Assets[:,t] += 1
            Returns[t] = num(1+R_p[t,-2])

        if pro == 'Non-Risky':
            # %60 Stocks ,%40 Bonds
            Wealth[t] =  num(1+1*BOND[t]+(1/12)*.02)*Wealth[t-1] #+0*R_p[t,-2]
            Assets[:,t] += 1
            Returns[t] = num(1+1*BOND[t])

            

        #i+=1
        #t+=1
    return Wealth,Assets,Returns


# In[ ]:





# In[ ]:





# In[ ]:





# In[151]:


idx_P = [0,1,2,4,6,8,10,12,14,16,17,18,20,26,27]
idx_PP = [0,1,2,4,6,8,10,12,14,16,17,18,20,21,26]
z1 = 169
BOND = num(R_t[z1:,21])
R_p = R_t[z1:,idx_P]
R_all = R_t[z1:,idx_P]
idxx = list(np.arange(28))
idxx[23] = 0
idxx[24] = 0
R_p_pred =y_pred_1[z1:,idx_PP]
R_pp = R_t[z1:,idx_PP]
R_pp[:,-2] += (1/12)*.02
#R_p_pred =y_pred_1[169:,idxx]
#R_all = R_t[169:,idxx]
EQ_P,Assets,Returns_P = generate_Wealth(pro = 'Non-Risky',R=R_p_pred)
EQ_M,Assets_M,Returns_M = generate_Wealth(pro = 'Model',R=R_p_pred)
EQ_SPY,Assets,Returns_Y = generate_Wealth(pro = 'SPY',R=R_p)
EQ_W,Assets,Returns_W = generate_Wealth(pro = 'Equal-Wieght',R=R_p)


# In[152]:


#np.max(Assets_M,0)
np.argmax(Assets_M[:,:],0)


# In[153]:


np.max(Assets_M[:,:],0)


# In[ ]:





# In[154]:


start = 0
EQ_M = generate_wealth(Returns_M,start)
EQ_SPY= generate_wealth(Returns_Y,start)
EQ_W  = generate_wealth(Returns_W,start)
EQ_P  = generate_wealth(Returns_P,start)


# In[155]:


z2 = -1
fig, ax = plt.subplots(figsize = (10,10))
ax.plot(Dates['XX'].iloc[(z1):],EQ_W[:],'tab:blue',label= 'Equal Wieght')
ax.plot(Dates['XX'].iloc[(z1):],EQ_M[:],'tab:red',label= 'Model')
#ax.plot(Dates['XX'].iloc[(t_0):],np.mean(EQ_M_sim,0),'tab:cyan')
ax.plot(Dates['XX'].iloc[(z1):],EQ_SPY[:],'tab:purple',label= 'SPY-Wealth')
ax.plot(Dates['XX'].iloc[(z1):],EQ_P[:],'tab:orange',label= 'Balanced')

#ax.plot(Dates['XX'].iloc[(t_0):],num(R_p[:,-2]),'tab:blue',label= 'SPY')
#ax.plot(Dates['XX'].iloc[(t_0):],num(R_p[:,0]),'tab:red',label= 'Taiwan')


for i, t in enumerate(ax.get_xticklabels()):
    if (i % 15) != 0 :
        t.set_visible(False)
ax.set_facecolor('peachpuff')
ax.grid()
ax.legend()


# In[83]:


print('Model Return:',12e0*np.mean( EQ_M[1:]/EQ_M[:-1]))
print('SPY Return:',12e0*np.mean(EQ_SPY[1:]/EQ_SPY[:-1])-1)
print('Equal Weight Return:',12e0*np.mean(EQ_W[1:]/EQ_W[:-1])-1)
print('Balanced Return:',12e0*np.mean(EQ_P[1:]/EQ_P[:-1])-1)


# In[84]:


print('Model variance:', 1e2*np.var(EQ_M[1:]/EQ_M[:-1]  )) 
print('SPY variance:', 1e2*np.var(EQ_SPY[1:]/EQ_SPY[:-1]))
print('Equal Weighted variance:', 1e2*np.var(EQ_W[1:]/EQ_W[:-1]))
print('Balanced variance:', 1e2*np.var(EQ_P[1:]/EQ_P[:-1]))


# In[85]:


print('Model Sharpe:',(12)*np.mean(EQ_M[1:]/EQ_M[:-1]-1)/np.std(EQ_M[1:]/EQ_M[:-1])  )
print('SPY Sharpe:',(12)*np.mean(EQ_SPY[1:]/EQ_SPY[:-1]-1)/np.std(EQ_SPY[1:]/EQ_SPY[:-1])  )
print('Equal Sharpe:',(12)*np.mean(EQ_W[1:]/EQ_W[:-1]-1)/np.std(EQ_W[1:]/EQ_W[:-1]))
print('Balanced Sharpe:', (12)*np.mean(EQ_P[1:]/EQ_P[:-1]-1)/np.std(EQ_P[1:]/EQ_P[:-1]))


# In[86]:


L_M = EQ_M[1:]/EQ_M[:-1]-1
L_SPY = EQ_SPY[1:]/EQ_SPY[:-1]-1
L_EQ = EQ_W[1:]/EQ_W[:-1]-1
L_EP = EQ_P[1:]/EQ_P[:-1]-1
loss = np.where(L_M<0 )[0]
print('Model Sortino:',(12)*np.std(L_M[loss])   )
loss = np.where(L_SPY<0 )[0]
print('SPY Sortino:',(12)*np.std(L_SPY[loss])     )
loss = np.where(L_EQ<0 )[0]
print('Equal Sortino:',(12)*np.std(L_EQ[loss])     )
loss = np.where(L_EP<0 )[0]
print('Balanced Sortino:', (12)*np.mean(EQ_P[1:]/EQ_P[:-1]-1)/np.std(EQ_P[1:]-EQ_P[:-1]))


# In[ ]:





# In[ ]:





# In[ ]:


#1e2*((52*2)*.0002)


# In[ ]:


#Dates


# In[ ]:


#np.sum(R_t[t_0:,26])
np.sum(R_p[:,:],0)


# In[ ]:


i


# In[ ]:





# In[ ]:


plt.plot(num(R_p[t_0:,0]))


# In[ ]:


np.sum( (1/15)*R_p[t_0:,])


# In[ ]:


np.sum(Wealth[:])


# In[ ]:


Dates['XX']


# In[ ]:


fig, ax = plt.subplots(figsize = (10,10))
ax.plot(Dates['XX'].iloc[(t_0-1):],Wealth,'tab:red',label = 'Portfolio')
ax.plot(Dates['XX'].iloc[(t_0):],num(R_t[t_0:,26]),'tab:blue',label = 'SPY')
#ax.plot(Dates['XX'].iloc[(t_0):],num(R_p[t_0:,1]),'tab:green')

#ax.plot(Dates['XX'].iloc[(t_0):],num(y_pred_1[t_0:,26]),'tab:purple',label = 'SPY-Forecast')

for i, t in enumerate(ax.get_xticklabels()):
    if (i % 25) != 0 :
        t.set_visible(False)
ax.set_facecolor('peachpuff')
ax.grid()
ax.legend()


# In[ ]:


R_p[t_0:,:].shape


# In[ ]:


t_0 = 169


# In[ ]:


Wealth[t] = np.sum(resh(W_P[:,t],[-1,1])*resh(num(R_p[200,:]),[-1,1]))


# In[ ]:


plt.plot(num(R_t[t_0:,0]))


# In[ ]:


np.sum(R_t[t_0:,0])


# In[ ]:


np.sum(R_t[:,0])


# In[ ]:


Dates['XX'].iloc[t_0]


# In[ ]:





# In[ ]:




