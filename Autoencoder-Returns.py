#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd
from pyspark.sql import SparkSession
import yfinance as yf
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:





# In[93]:


# Code in file nn/two_layer_net_optim.py
import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 5, 1000, 100, 64

# Create random Tensors to hold inputs and outputs.
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
        )
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algorithms. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
  # Forward pass: compute predicted y by passing x to the model.
  y_pred = model(x)

  # Compute and print loss.
  loss = loss_fn(y_pred, y+1)
  #print(t, loss.item())
  
  # Before the backward pass, use the optimizer object to zero all of the
  # gradients for the Tensors it will update (which are the learnable weights
  # of the model)
  optimizer.zero_grad()

  # Backward pass: compute gradient of the loss with respect to model parameters
  loss.backward()

  # Calling the step function on an Optimizer makes an update to its parameters
  optimizer.step()


# In[ ]:





# In[94]:


m_1 = 3;    #Equity 1
m_2 = 6;    #Equity 2
m_3 = 8;    #Equity 3
m_4 = 11;   #Equity 4
m_5 = 14;   #Equity 5
m_6 = 18;   #Equity 6
m_7 = 22;   #Equity 7
m_8 = 26;   #Equity 8
m_9 = 30;   #Equity 9
m_10 = 33;  #Equity 10
m_11 = 36;  #Equity 11
m_12 = 41;  #Equity 12
m_13 = 43;  #Equity 13
m_14 = 44;  #Equity 14  
m_15 = 44;  #Equity 15
m_16 = 45;  #Equity 16
m_17 = 49;  #Equity 17
m_18 = 52;  #Equity 18
r_1 = 0     #Equity 19
r_2 = 1     #Equity 20
r_3 = 4     #Equity 21
r_4 = 6     #Equity 22
r_5 = 8     #Equity 23
r_6 = 10    #Equity 24
r_7 = 12    #Equity 25
r_8 = 14    #Equity 26 
r_9 = 16    #Equity 27 
r_10 = 16   #Equity 28 
r_11 = 17   #Equity 29 
r_12 = 20   #Equity 30
r_13 = 20   #Equity 31   
r_14 = 21   #Equity 32
r_15 = 22   #Equity 33
r_16 = 23   #Equity 34
r_17 = 24   #Equity 35
r_18 = 25   #Bond   1
r_19 = 26   #Bond   2  
r_20 = 27   #Bond   3


# In[102]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z1 = 0
z2 = 2
X_t = (np.load('/home/ajay/Downloads/Threshold Model/X_t.npy'))
R_t = (np.load('/home/ajay/Downloads/Threshold Model/R_t.npy'))
n =int(R_t.shape[1])
T  =R_t.shape[0]
n_x = X_t.shape[0]
X_t = (X_t-np.reshape(np.mean(X_t,0),[1,-1]))/np.reshape(np.std(X_t,0),[1,-1])
y = torch.reshape(torch.from_numpy(X_t).float(),[T,53])
r = torch.reshape(torch.from_numpy(R_t).float(),[T,n])


# In[103]:


class Network(nn.Module):
    def __init__(self,z):
        D_out = z.shape[1]
        H  = 1
        super(Network, self).__init__()
        
        self.encode = nn.Sequential(
          #nn.Linear(D_out,2*H),
          nn.Linear(D_out,2*H),
          nn.Linear(2*H,2*H),
          nn.Sigmoid(),
          nn.Linear(2*H, H),


        )
        self.decode = nn.Sequential(
          nn.Linear(H, D_out)  
        )
  
    def forward(self, x):
        # Forward pass 
        encode = self.encode(x)
        y_pred = self.decode(encode)
        return y_pred,encode
    def SSE(self, x1,x2):
        loss = x1-x2
        #loss[torch.where(torch.isnan(loss))[0],torch.where(torch.isnan(loss))[1]] = 0
        loss =torch.sum(loss**2)
        return loss
        
        
        


# In[ ]:





# In[104]:



model = Network(y[0:150,0:3])
y_pred,encode = model.forward(y[0:150,0:3])
#model.SSE(y_pred,y[0:150,0:3])


# In[105]:



model.cuda()
torch.cuda.is_available()
torch.cuda.get_device_name(0)


# In[106]:


t2 = 250;m_2=0;m_3=3


# In[107]:


y_pred,encode = model.forward(y[0:t2,0:3].to(device))
loss =model.SSE(y_pred.to(device), y[0:t2,0:3].to(device)) 
print(loss)


# In[ ]:





# In[ ]:


learning_rate = 1e-2
#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(15000):
  y_pred,encode = model.forward(y[0:t2,0:3].to(device))
  loss =model.SSE(y_pred.to(device), y[0:t2,0:3].to(device)) 
  if t%1000==0:
      print(t, loss.item())  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


# In[ ]:


y_pred,encode = model.forward(y[:,0:3].to(device))
loss =model.SSE(y_pred.to(device),y[:,0:3].to(device)) 
print(loss)


# In[ ]:





# In[ ]:





# In[39]:





# In[ ]:





# In[ ]:





# In[109]:


def Train(z2):
    model = Network(z2)
    y_pred,encode = model.forward(z2)
    model.SSE(y_pred,z2)
    model.cuda()
    
    learning_rate = 1e-1;t2 = 250
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(15000):
      #print(t)
      y_pred,encode = model.forward(z2[0:t2,:].to(device))
      loss = model.SSE(y_pred,z2[0:t2,:].to(device))
      optimizer.zero_grad()

    y_pred,f  = model.forward(z2.to(device))
    errs = model.SSE(y_pred,z2.to(device))

    return f,errs,y_pred



# In[110]:


Model_list = np.array([
          m_1,m_2,m_3,
          m_4,m_5,m_6,
          m_7,m_8,m_9,
          m_10,m_11,m_12
         
                      ])
m_1 = 4;    #Country 1
m_2 = 7;    #Country 2
m_3 = 10;   #Country 3
m_4 = 13;   #Country 4
m_5 = 16;   #Country 5
m_6 = 20;   #Country 6
m_7 = 24;   #Country 7
m_8 = 28;   #Country 8
m_9 = 32;   #Country 9 
m_10 = 35;  #Country 10
m_11 = 38;  #Country 11
m_12 = 42;  #Country 12
m_13 = 44;  #Country 13   
m_14 = 44;  #Country 14  
m_15 = 44;  #Country 15
m_16 = 45;  #Country 16
m_17 = 50;  #Country 17
m_18 = 52;  #Country 18
np.arange(100)[0:Model_list[0]]
np.arange(100)[Model_list[0]:Model_list[1]]
np.arange(100)[Model_list[1]:Model_list[2]]

### Load the Data ###
F = torch.zeros([X_t.shape[0],1,14])
ERRS = torch.zeros([14])
for i in range(12):
    print(i)
    if i==0:
        F[:,:,i],ERRS[i],y_pred = Train(y[:,0:m_1])
    else:
        F[:,:,i],ERRS[i],y_pred = Train(y[:,Model_list[i-1]:Model_list[i]])
    


# In[111]:


X_t.shape[0]


# In[112]:


###Thailand### 
x1 = torch.arange(100)[47:m_17]
x2 = torch.arange(100)[m_12:m_13]
px = torch.cat((x2, x1), dim=0)
F[:,:,-2],ERRS[-2],y_pred = Train(y[:,px]) 
###Taiwan### 
x1 = torch.arange(100)[50:53]
x2 = torch.arange(100)[(m_14):(m_14+1)]
px = torch.cat((x2, x1), dim=0)
F[:,:,-1],ERRS[-1],y_pred = Train(y[:,px]) 


# In[113]:


x1 = torch.arange(100)[50:53]
x2 = torch.arange(100)[(m_14):(m_14+1)]
px = torch.cat((x2, x1), dim=0)
F[:,:,-1],ERRS[-1],y_pred = Train(y[:,px]) 


# In[114]:


#plt.plot(torch.reshape(F,[T,-1]).cpu().detach().numpy())
F.shape


# In[ ]:





# In[115]:


F.shape
F[:,:,11]


# In[82]:


F.shape


# In[ ]:


#plt.plot(np.reshape(F.cpu().detach().numpy(),[-1,14])[:,:])


# In[116]:


np.save('/home/ajay/Downloads/Threshold Model/f2_t.npy',np.reshape(F.cpu().detach().numpy(),[-1,14]))


# In[87]:


np.load('/home/ajay/Downloads/Threshold Model/R_t.npy').shape


# In[89]:


np.load('/home/ajay/Downloads/Threshold Model/f2_t.npy').shape


# In[ ]:




