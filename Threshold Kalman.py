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
    P = 2000
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