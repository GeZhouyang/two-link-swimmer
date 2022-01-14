""" A little code that uses Q-learning, a form of reinforcement learning,
    to teach a simple two-link microrobot to swim in viscous fluids. 

    See the following references for more physical details:
    [1] Alan Tsang et al. Phys. Rev. Fluids (2020),
    [2] Nijafi and Golestanian, Phys. Rev. E (2004).

    Author: Anthony Ge
    Date: Jan 12, 2022
"""


import os
import sys
import numpy as np
import random
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.use('Agg')

from datetime import datetime
print(datetime.now())


#### model parameters

alpha = 0.5   # learning rate [0,1]
gamma = 0.8   # belief in future [0,1)
epsln = 0.1   # exploration factor [0,1)

print('Learning rate %.1f,  belief in future %.1f,  tendency to explore %.1f'
      %(alpha,gamma,epsln))

N_max = 200
s_tr = list(range(N_max))  # training steps

## Index mapping from state and action to reward and Q:
## --------------------------
##      State       | Action
## --------------------------
## ind_s  (s_l,s_r) |  L  R
## --------------------------
##   1  :=  (0,0)   |  0  1
##   2  :=  (0,1)   |  2  3  <-- qval/reward index
##   3  :=  (1,0)   |  4  5
##   4  :=  (1,1)   |  6  7 
## --------------------------
## Pre-setting the reward array (equivalent to the center-of-mass displacement)
#  First, tabulate the center particle displacement, see Ref.[2]
r1, r2 = 1.35, 1.44  
dx1 = [-r1,r1, -r2,-r1, r1,r2, r2,-r2]
#  Second, tabulate the reward (-4/3 is the reverse disp. of the center of mass)
r1 -= 4./3
r2 -= 4./3.  
rwd = [-r1,r1, -r2,-r1, r1,r2, r2,-r2]
#print(rwd)
#sys.exit()

## Function mapping configuration to state index
def state_index(s_l, s_r):
    return 2*s_l + s_r + 1

## Save a snapshot
def snapshot(istep,x1,s_l,s_r):
    ## plot in figures
    fig, ax = plt.subplots(1,1, sharex=True)
    fig.set_size_inches(8,2)
    ## spheres
    if s_l == 0:
        x2 = x1 - 10
    if s_l == 1:
        x2 = x1 - 6
    if s_r == 0:
        x3 = x1 + 10
    if s_r == 1:
        x3 = x1 + 6
    c1 = plt.Circle((x1,0),1, fill=False)
    c2 = plt.Circle((x2,0),1, fill=False)
    c3 = plt.Circle((x3,0),1, fill=False)
    ax.add_artist(c1) 
    ax.add_artist(c2) 
    ax.add_artist(c3) 
    ## links
    ax.plot([x2,x3],[0,0],':',c='k',alpha=0.5)
    ## center of mass
    xc = (x1+x2+x3)/3.
    ax.plot(xc,0,'o',c='C7')  
    
    ax.text(0.06, 0.11,('n = %4i'%istep),
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax.transAxes)

    ax.set_xlim(-15,15)
    ax.set_ylim(-2,2)

    ax.set_xlabel(r'$x/R$')
    ax.set_ylabel(r'$y/R$')

    ax.set_aspect('equal')
    
    plt.savefig('pics/snap'+('%05i'%istep)+'.png', bbox_inches='tight')
    plt.close()

    return

if __name__ == '__main__':
         
    ## Initialization
    pos1 = 0.           # initial position of the center particle
    s_l, s_r  = 0, 0    # initial state, left and right
    ind_s = state_index(s_l,s_r)
    qval = np.zeros(8)  # the Q-values array
    disp = 0.           # net displacement of the swimmer
    disp_l = [disp]
    qv1 = [0.]  ###check qv1
    
    #snapshot(0,pos1,s_l,s_r)  # save a snapshot of the configuration
    
    ## Q-learning
    for istep in s_tr:
        
        ## Query the action-value at present
        i0 = 2*(ind_s-1)
        i1 = i0+2
        qq = qval[i0:i1]
        rr = rwd[ i0:i1]  # the possible rewards 

        ## Take an action (stochastic policy)
        num = random.uniform(0,1)
        flag = ''
        if num > epsln:
            ## Act according to the present Q-values
            if qq[0] > qq[1]: 
                s_l = 1 - s_l        # flip the left link (L)
                i_r = 2*(ind_s-1)    # index into the reward/Q array
            else:
                s_r = 1 - s_r        # ........ right ... (R)
                i_r = 2*(ind_s-1)+1  # index into the reward/Q array
        else:
            ## Explore a random action
            flag = '(exploring)'
            num1 = random.uniform(0,1)
            if num1 < 0.5:
                s_l = 1 - s_l        # flip the left link (L)
                i_r = 2*(ind_s-1)    # index into the reward/Q array
            else:
                s_r = 1 - s_r        # ........ right ... (R)
                i_r = 2*(ind_s-1)+1  # index into the reward/Q array

        ## Update the state
        ind_s = state_index(s_l,s_r)
        disp += rwd[i_r]  # the net displacement
        disp_l.append(disp)
        pos1 += dx1[i_r]  # displacement of the center particle
        #snapshot(istep+1,pos1,s_l,s_r)  # save a snapshot of the configuration
        
        ## Receive an immediate payoff
        payoff = rwd[i_r]

        ## Estimated the future gain based on the present knowledge
        i0 = 2*(ind_s-1)
        i1 = i0+2
        qq = qval[i0:i1]
        ff = max(qq)  # foresight

        ## Adjust the Q-value
        qval[i_r] = (1.-alpha)*qval[i_r] + alpha*(payoff + gamma*ff)
        qv1.append(qval[0])  ###check qv1
        
        print('Training step %5i,  net disp. %.2f  %s'%(istep,disp,flag))
        
    print(qval)

    s_tr.append(istep+1)
    s_tr   = np.asarray(s_tr)
    disp_l = np.asarray(disp_l)
    
    ## plot in figures
    fig, ax = plt.subplots(1,1, sharex=True)
    fig.set_size_inches(3.5,3.5)

    ## Trajectory of the learning agent (center of mass)
    ax.plot(s_tr, disp_l)

    ## Ideal solution: a travelling wave
    xx = np.linspace(0,N_max,2)
    yy = 0.045*xx
    ax.plot(xx,yy,'--', c='C7', alpha=0.6)
    
    ax.set_xlim(0,N_max)

    ax.set_xlabel(r'$n$')
    ax.set_ylabel(r'$\Delta/R$')
    
    plt.savefig('curve_disp.pdf', bbox_inches='tight')
    plt.close()

    ## plot in figures
    fig, ax = plt.subplots(1,1, sharex=True)
    fig.set_size_inches(3.5,5)

    table = qval.reshape((4,2))

    plt.imshow(table, cmap='Greys')

    row_labels = ['(0,0)','(0,1)','(1,0)','(1,1)']
    col_labels = ['L','R']

    plt.xticks(range(2), col_labels)
    plt.yticks(range(4), row_labels)

    ax.set_xlabel(r'Action')
    ax.set_ylabel(r'State')
    
    plt.savefig('table_Q.pdf', bbox_inches='tight')
    plt.close()


    ## plot in figures
    fig, ax = plt.subplots(1,1, sharex=True)
    fig.set_size_inches(3.5,3.5)

    ## check the evolution of the first entry in Q
    ax.plot(s_tr, qv1)
    
    ax.set_xlim(0,N_max)

    ax.set_xlabel(r'$n$')
    ax.set_ylabel(r'$Q_1$')
    
    plt.savefig('curve_q1.pdf', bbox_inches='tight')
    plt.close()
