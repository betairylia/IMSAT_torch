#import libraries
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

import contextlib


from scipy.sparse import csr_matrix, triu
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
from scipy.sparse import identity
from sklearn.neighbors import NearestNeighbors

import time

from munkres import Munkres

from datasets import *
import argparse

parser = argparse.ArgumentParser(description='MIST experiments')

# Set-up
parser.add_argument('--dataset', type=str, default='mnist')

# General HP
parser.add_argument('--batchsize', type=int, default=250, help='batchsize')
parser.add_argument('--epochs', type=int, default=50, help='epoch')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')

# HP
parser.add_argument('--xi', type=float, default=10.0, help='xi (VAT)')
parser.add_argument('--mu1', type=float, default=0.4, help='xi (VAT)')

args = parser.parse_args()

# by using below command, gpu is available
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

X, Y, _, dim, C = GetData(args.dataset)

def conditional_entropy(soft):
    loss = torch.sum(-soft*torch.log(soft + 1e-8)) / soft.shape[0]
    return loss


def entropy(soft):
    avg_soft = torch.mean(soft, 0, True) 
    loss = -torch.sum(avg_soft * torch.log(avg_soft + 1e-8))
    return loss


def kl(p, q):
    loss = torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8))) / p.shape[0]
    return loss






@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def return_vat_Loss(model, x, xi, eps):

    optimizer.zero_grad()
    
    with _disable_tracking_bn_stats(model):
        with torch.no_grad():
            target = torch.softmax(model(x), 1) 
        
        d = torch.randn(x.shape).to(dev)
        d = _l2_normalize(d)
        d.requires_grad_()
        out_vadv = model(x + xi*d)
        hat = torch.softmax(out_vadv, 1)
        adv_distance = kl(target, hat)

        adv_distance.backward()
        
        d = _l2_normalize(d.grad)
        r_adv = eps * d
        out_vadv = model(x + r_adv)
        hat = torch.softmax(out_vadv, 1)
        R_vat = kl(target, hat)

    return R_vat





def ReturnACC(cluster, target_cluster, k):
    """ Compute error between cluster and target cluster
    :param cluster: proposed cluster
    :param target_cluster: target cluster
    k: number of classes
    :return: error
    """
    n = np.shape(target_cluster)[0]
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            M[i][j] = np.sum(np.logical_and(cluster == i, target_cluster == j))
    m = Munkres()
    indexes = m.compute(-M)
    corresp = []
    for i in range(k):
        corresp.append(indexes[i][1])
    pred_corresp = [corresp[int(predicted)] for predicted in cluster]
    acc = np.sum(pred_corresp == target_cluster) / float(len(target_cluster))
    return acc 

import os

start = time.time()

alpha = 0.25
K = 10 # Number of neighbors

knncachestr = "%s-k%d.npy" % (args.dataset, K+1)
  
if args.dataset != "unknown" and os.path.exists(knncachestr):
    
  print("Loaded cached kNN from %s" % knncachestr)
  distances = np.load(knncachestr)
  
else:

  nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='brute').fit(X)
  distances, indices = nbrs.kneighbors(X)
  np.save(knncachestr, distances)
  #del _

R = alpha*distances[:,K]
R = R.reshape(X.shape[0],1)

del distances

end = time.time()
print(f"{end-start} seconds by Brute-Force KNN.")


#plt.hist(R, bins=100)

R = torch.tensor(R.astype('f')).to(dev)


X = torch.tensor(X.astype('f')).to(dev) # this unlabeled dataset (set of feature vectors) is input of IMSAT

# define archtechture of MLP(Multi Layer Perceptron). 
# in this net, batch-normalization (bn) is used. 
# bn is very important to stabilize the training of net. 
#torch.manual_seed(0)
class Net(nn.Module): 

    def __init__(self):
        super(Net, self).__init__()

        self.l1 = nn.Linear(dim, 1200)
        self.bn1 = nn.BatchNorm1d(1200)

        self.l2 = nn.Linear(1200,1200)
        self.bn2 = nn.BatchNorm1d(1200)

        self.l3 = nn.Linear(1200,C)
        
    def forward(self, x):
       
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        x = self.l3(x)
        
        return x




net = Net()

# throw net object to gpu
net = net.to(dev)


################################################## Training of imsat ############################################

# decide hyperparameter values for imsat training
epochs = args.epochs # number of epochs

xi = args.xi       # xi is used for making adversarial vector. if xi becomes smaller, 
                   # theoretically obtained r_vadv becomes more priecise


mini_size = args.batchsize # size of mini-batch training dataset

m = X.shape[0]//mini_size # number of iteration at each epoch 

mu1 = args.mu1 # regularyzing constant for H(y) #mu1
mu2 = mu1/4 # regularyzing constant for H(y|x) #mu2


## define optimizer for set of parameters in deep neural network
## lr is the learning rate of parameter vector, betas are the lr of gradient and second moment of gradient
optimizer = optim.Adam(net.parameters(), 
                        lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)




print("Start training of IMSAT.")
for epoch in range(epochs):
    print("At ", epoch, "-th epoch, ")

    # set empiricial loss to 0 in the beginning of epoch
    empirical_loss = 0.0

    idx_eph = np.random.permutation(X.shape[0])
    
    net.train()

    for itr in range(m):
      
      ## chose a core idx of mini_batch
      idx_itr = idx_eph[itr*mini_size:(itr+1)*mini_size]

      ## define components at each iteration
      X_itr = X[idx_itr,:]
      R_itr = R[idx_itr,:]

            
      R_vat = return_vat_Loss(net, X_itr, xi, R_itr)

      out_itr = net(X_itr) 
      soft_out_itr = torch.softmax(out_itr, 1)
      


      ## define class-balance loss via H(p(y)). H(p(y)) should be maximized.
      cb_loss = -entropy(soft_out_itr)

      ## define shannon conditional entropy loss H(p(y|x)) named by c_Ent.
      c_Ent = conditional_entropy(soft_out_itr)
              

      # objective of imsat
      objective = R_vat + mu1*cb_loss + mu2*c_Ent
    

      # update the set of parameters in deep neural network by minimizing loss
      optimizer.zero_grad() 
      objective.backward()
      optimizer.step()

      empirical_loss = empirical_loss + objective.data


    #empirical_loss = running_loss/m
    empirical_loss = empirical_loss.cpu().numpy()
    print("average empirical loss is", empirical_loss/m, ',')

    net.eval()

    # at each epoch, prediction accuracy is displayed
    with torch.no_grad():
        out = net(X)
        preds = torch.argmax(out, dim=1)
        preds = preds.cpu().numpy()
        preds = preds.reshape(1, preds.shape[0])
        clustering_acc = ReturnACC(preds[0], Y[0], C)
    
    print("and current clustering accuracy is", clustering_acc )
