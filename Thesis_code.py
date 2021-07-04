# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:01:45 2021

@author: Soham Bandyopadhyay
"""

import numpy as np
import networkx as nx
import math
import statistics
from matplotlib import pyplot as plt
from itertools import combinations 
from scipy import stats
rnd = np.random


def gini(array):
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element    
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient


r = 2 #synergy factor
p0 = 0.73 #environmental influence
lamb = 0.001 #influence of wealth
b = 0.1 #bias term
pc_frac = 0.0 #fraction of punishers
print("**** Punisher Fraction***** ", pc_frac)
print("*******p0******", p0)
print("*****lambda*****", lamb)
pun_cost = 10 #cost of punishment
pun_fine = 100 #punishment fine
nrounds = 50 #number of rounds

re = 0.0 #rewiring fraction
print("**** Rewiring Fraction***** ", re)
pr = 0.87 #retaining existing link
pb = 0.7 #breaking existing link
pm = 0.93 #making new link when both coop
pe = 0.3 #making new link when one cooperates
ps = 0.2 #making new link when neither cooperate


sumwealth = np.empty(100)

def_errors = 0
coop_errors = 0

c_envlist = np.zeros([(nrounds-1),100])
s_envlist = np.zeros([(nrounds-1),100])
n_envlist = np.zeros([(nrounds-1),100])

r_cooplist = np.zeros([nrounds,100])
p_cooplist = np.zeros([nrounds,100])
r_deflist = np.zeros([nrounds,100])
p_deflist = np.zeros([nrounds,100])

deg_coop = np.zeros([nrounds,100])
deg_def = np.zeros([nrounds,100])

deg_coop_prev = np.zeros([(nrounds-1),100])
deg_def_prev = np.zeros([(nrounds-1),100])

gini_list = np.empty([(nrounds+1),100])

coop_neighbourlist = np.zeros([nrounds,100])
def_neighbourlist = np.zeros([nrounds,100])

pcdifflist = []
rcdifflist = []
pddifflist = []
rddifflist = []

gincof = np.empty(100)
coop_num = np.empty(100)

error_count=0
initgini = 0


avg_stratlist = []
final_wealthlist = []
colour_list = []
deglist_small = np.empty(50)
init_deglist = np.empty(50)
deglist_large = []
degchange_large = []
final_stratlist_large = []

m = np.zeros(100)

for m_ctr in range(100):
    
    G = nx.gnp_random_graph(50,0.3)
        
    nodelist = np.array(G.nodes)
    wealthlist = np.empty(50)
    last_stratlist = np.empty(50)
    ind_stratlist = np.zeros([50,nrounds])
    
    for i in range(50):
        init_deglist[i] = G.degree[i]
    
    
    
    for i in range(50):
        x = rnd.rand()
        if(x < 0.5):
            wealthlist[i] = 3000
        elif(x >= 0.5):
            wealthlist[i] = 7000
    
    temp_wealthlist = wealthlist.copy()
    wealthlist_g = wealthlist.copy()
    init_wealthlist = wealthlist.copy()
    gini_list[0,m_ctr] = gini(wealthlist_g)
    initgini += gini(wealthlist_g)
    
    for i in range(50):
        nbrlist = np.array(G.adj[i])
        deg = G.degree[i]
        x1 = rnd.rand()
        if(x1<0.7):
            last_stratlist[i] = 0
            for j in nbrlist:
                temp_wealthlist[j] += r*50 
            temp_wealthlist[i] -= deg*50
        elif(x1>=0.7):
            last_stratlist[i] = 1
        ind_stratlist[i,0] = last_stratlist[i]
            
            
    defsum = 0
    degsum = 0
    for i in range(50):
        nbrlist = np.array(G.adj[i])
        degsum += G.degree[i]
        for k in nbrlist:
            defsum += last_stratlist[k]
        if (last_stratlist[i]==0):
            x3 = rnd.rand()
            if (x3<pc_frac):                
                for j in nbrlist:
                    if (last_stratlist[j]==1):
                        temp_wealthlist[j] -= pun_fine
                        temp_wealthlist[i] -= pun_cost
    coopsum = degsum-defsum
    def_neighbourlist[0,m_ctr] = defsum/50
    coop_neighbourlist[0,m_ctr] = coopsum/50
                        
    for i in range(50):
        if (temp_wealthlist[i]<=0):
            error_count +=1
            print("error", error_count)
            print(temp_wealthlist[i])
            if last_stratlist[i]==0:
                coop_errors +=1
            elif last_stratlist[i]==1:
                def_errors +=1
            
    r_coop = 0
    p_coop = 0
    r_def = 0
    p_def = 0
    for i in range(50):
        wealthsum = 0
        nbrlist = np.array(G.adj[i])
        deg = G.degree[i]
        for l in nbrlist:
            wealthsum += wealthlist[l]
        wealth_avg = wealthsum/deg
        wealth_diff = wealth_avg - wealthlist[i]
        if(wealth_diff<0) and (last_stratlist[i]==0):
            r_coop +=1
            deg_coop[0,m_ctr] += deg
        elif(wealth_diff>0) and (last_stratlist[i]==0):
            p_coop +=1
            deg_coop[0,m_ctr] += deg
        elif(wealth_diff<0) and (last_stratlist[i]==1):
            r_def +=1
            deg_def[0,m_ctr] += deg
        elif(wealth_diff>0) and (last_stratlist[i]==1):
            p_def +=1
            deg_def[0,m_ctr] += deg
            
    r_cooplist[0,m_ctr] = r_coop
    p_cooplist[0,m_ctr] = p_coop
    r_deflist[0,m_ctr] = r_def
    p_deflist[0,m_ctr] = p_def
    
    deg_coop[0,m_ctr] /= (p_coop + r_coop)
    deg_def[0,m_ctr] /= (p_def + r_def)
    
    pairs = list(combinations(nodelist, 2))
    for i in pairs: 
        rex1 = rnd.rand()
        if(rex1<re):
           edge_exists = G.has_edge(*i)
           if edge_exists == True:
               playerlink = rnd.choice(i)
               if last_stratlist[playerlink] == 0:
                   rex2 = rnd.rand()
                   if (rex2>=pr):
                       G.remove_edge(*i)
               elif last_stratlist[playerlink] == 1:
                   rex2 = rnd.rand()
                   if (rex2<pb):
                       G.remove_edge(*i)
           elif edge_exists == False:
               make_edge = last_stratlist[i[0]] + last_stratlist[i[1]]
               rex3 = rnd.rand()
               if make_edge == 0:
                   if (rex3<pm):
                       G.add_edge(*i)
               elif make_edge == 1:
                   if (rex3<pe):
                       G.add_edge(*i)
               elif make_edge == 2:
                   if (rex3<ps):
                       G.add_edge(*i)
               
                   
    wealthlist = temp_wealthlist
    temp_wealthlist = wealthlist.copy()
    wealthlist_g = wealthlist.copy()
    gini_list[1,m_ctr] = gini(wealthlist_g)
    stratlist = last_stratlist.copy()
    
    for ctr in range((nrounds-1)):
        c_env = 0
        s_env = 0
        n_env = 0
        r_coop = 0
        p_coop = 0
        r_def = 0
        p_def = 0
        ncoop_prev = 0
        ndef_prev = 0
        for i in range(50):
            stratsum = 0
            wealthsum = 0
            nbrlist = np.array(G.adj[i])
            deg = G.degree[i]
            for k in nbrlist:
                stratsum += last_stratlist[k]
            
            if last_stratlist[i] == 0:
                deg_coop_prev[ctr,m_ctr] += deg
                ncoop_prev += 1
            elif last_stratlist[i] == 1:
                deg_def_prev[ctr,m_ctr] += deg
                ndef_prev += 1
                
            for l in nbrlist:
                wealthsum += wealthlist[l]
            wealth_avg = wealthsum/deg
            wealth_diff = wealth_avg - wealthlist[i]
                        
            
            if(stratsum/deg)<= 0.5:
                if(stratsum/deg)< 0.5:
                    c_env +=1
                elif(stratsum/deg)== 0.5:
                    n_env +=1
                pc = p0 - b + ((1-p0)*(math.tanh(lamb*wealth_diff)))
                if pc>1:
                    print("error2")
                x2 = rnd.rand()
                if(x2<pc):
                    stratlist[i] = 0
                    deg_coop[ctr+1,m_ctr] += deg
                    if(wealth_diff<=0):
                        r_coop +=1
                        if(ctr == (nrounds-2)):
                            rcdifflist.append(math.tanh(lamb*wealth_diff))
                    elif(wealth_diff>0):
                        p_coop +=1
                        if(ctr == (nrounds-2)):
                            pcdifflist.append(math.tanh(lamb*wealth_diff))
                    for j in nbrlist:
                        temp_wealthlist[j] += r*50 
                    temp_wealthlist[i] -= deg*50
                elif(x2>=pc):
                    if(wealth_diff<=0):
                        r_def +=1
                        if(ctr == (nrounds-2)):
                            rddifflist.append(math.tanh(lamb*wealth_diff))
                    elif(wealth_diff>0):
                        p_def +=1
                        if(ctr == (nrounds-2)):
                            pddifflist.append(math.tanh(lamb*wealth_diff))
                    stratlist[i] = 1
                    deg_def[ctr+1,m_ctr] += deg
                    
            elif(stratsum/deg)>0.5:
                s_env+= 1
                pc = (1 - p0 + b) - ((1-p0)*(math.tanh(lamb*wealth_diff)))
                if pc>1:
                    print("error2")
                x2 = rnd.rand()
                if(x2<pc):
                    stratlist[i] = 0
                    deg_coop[ctr+1,m_ctr] += deg
                    if(wealth_diff<=0):
                        r_coop +=1
                        if(ctr == (nrounds-2)):
                            rcdifflist.append(math.tanh(lamb*wealth_diff))
                    elif(wealth_diff>0):
                        p_coop +=1
                        if(ctr == (nrounds-2)):
                            pcdifflist.append(math.tanh(lamb*wealth_diff))
                    for j in nbrlist:
                        temp_wealthlist[j] += r*50 
                    temp_wealthlist[i] -= deg*50
                elif(x2>=pc):
                    if(wealth_diff<=0):
                        r_def +=1
                        if(ctr == (nrounds-2)):
                            rddifflist.append(math.tanh(lamb*wealth_diff))
                    elif(wealth_diff>0):
                        p_def +=1
                        if(ctr == (nrounds-2)):
                            pddifflist.append(math.tanh(lamb*wealth_diff))
                    stratlist[i] = 1
                    deg_def[ctr+1,m_ctr] += deg
                    
            ind_stratlist[i,ctr+1] = stratlist[i]
            
    
        defsum = 0  
        degsum = 0
        for i in range(50):
            nbrlist = np.array(G.adj[i])
            degsum += G.degree[i]
            for k in nbrlist:
                defsum += stratlist[k]                
            if (stratlist[i]==0):
                x4 = rnd.rand()
                if (x4<pc_frac):                    
                    for j in nbrlist:
                        if (stratlist[j]==1):
                            temp_wealthlist[j] -= pun_fine
                            temp_wealthlist[i] -= pun_cost 
        coopsum = degsum-defsum
        def_neighbourlist[ctr+1,m_ctr] = defsum/50
        coop_neighbourlist[ctr+1,m_ctr] = coopsum/50    

                
                    
        
        for i in range(50):
            if (temp_wealthlist[i]<=0):
                error_count +=1
                print("error", error_count)
                print(temp_wealthlist[i], ctr, stratlist[i])
                if stratlist[i]==0:
                    coop_errors +=1
                elif stratlist[i]==1:
                    def_errors +=1
        
        
        c_envlist[ctr,m_ctr] = c_env
        s_envlist[ctr,m_ctr] = s_env
        n_envlist[ctr,m_ctr] = n_env  
        
        r_cooplist[ctr+1,m_ctr] = r_coop
        p_cooplist[ctr+1,m_ctr] = p_coop
        r_deflist[ctr+1,m_ctr] = r_def
        p_deflist[ctr+1,m_ctr] = p_def     
        
        deg_coop[ctr+1,m_ctr] /= (p_coop + r_coop)
        deg_def[ctr+1,m_ctr] /= (p_def + r_def)
        
        deg_coop_prev[ctr,m_ctr] /= ncoop_prev
        deg_def_prev[ctr,m_ctr] /= ndef_prev


        pairs = list(combinations(nodelist, 2))
        for i in pairs: 
            rex1 = rnd.rand()
            if(rex1<re):
               edge_exists = G.has_edge(*i)
               if edge_exists == True:
                   playerlink = rnd.choice(i)
                   if stratlist[playerlink] == 0:
                       rex2 = rnd.rand()
                       if (rex2>=pr):
                           G.remove_edge(*i)
                   elif stratlist[playerlink] == 1:
                       rex2 = rnd.rand()
                       if (rex2<pb):
                           G.remove_edge(*i)
               elif edge_exists == False:
                   make_edge = stratlist[i[0]] + stratlist[i[1]]
                   rex3 = rnd.rand()
                   if make_edge == 0:
                       if (rex3<pm):
                           G.add_edge(*i)
                   elif make_edge == 1:
                       if (rex3<pe):
                           G.add_edge(*i)
                   elif make_edge == 2:
                       if (rex3<ps):
                           G.add_edge(*i)
                    
        
        last_stratlist = stratlist  
        stratlist = last_stratlist.copy()          
        wealthlist = temp_wealthlist
        wealthlist_g = wealthlist.copy()
        gini_list[(ctr+2),m_ctr] = gini(wealthlist_g)
        temp_wealthlist = wealthlist.copy()
        
    sumwealth[m_ctr] = np.sum(wealthlist)
    
    for i in range(50):
        deglist_small[i] = G.degree[i]
        
        
    sum_stratlist = np.sum(ind_stratlist, axis=1)
    degchange = deglist_small - init_deglist
    wealthchange = wealthlist - init_wealthlist
    col = np.empty(50, dtype=str)
    for i in range(50):
        if (init_wealthlist[i]==3000):
            col[i] = 'r'
        elif (init_wealthlist[i]==7000):
            col[i] = 'b'
    
    final_stratlist_large += stratlist.tolist()
    deglist_large += deglist_small.tolist()
    degchange_large += degchange.tolist()
    avg_stratlist += sum_stratlist.tolist()
    final_wealthlist += wealthchange.tolist()
    colour_list += col.tolist()
    
    for i in wealthlist:
        if i>np.mean(wealthlist):
            m[m_ctr] += 1

    gincof[m_ctr] = gini(wealthlist) 
    
    def_num = np.sum(last_stratlist)
    coop_num[m_ctr] = (50 - def_num)/50
    
normal = np.around(gincof, 2)
(unique, counts) = np.unique(normal, return_counts=True)
frequencies = np.asarray((unique, counts))
    
   
print("rich ", np.mean(m), " poor ", 50 - np.mean(m), " +- ", 2*(np.std(m))/10)

print("Average total wealth after ",nrounds," rounds = ", np.mean(sumwealth), "+- ", 2*(np.std(sumwealth))/10)

c_mean = np.mean(c_envlist, axis = 1)
c_std = np.std(c_envlist, axis = 1)
c_stdplus = c_mean + 2*(c_std/10)
c_stdminus = c_mean - 2*(c_std/10)

s_mean = np.mean(s_envlist, axis = 1)
s_std = np.std(s_envlist, axis = 1)
s_stdplus = s_mean + 2*(s_std/10)
s_stdminus = s_mean - 2*(s_std/10)

n_mean = np.mean(n_envlist, axis = 1)
n_std = np.std(n_envlist, axis = 1)
n_stdplus = n_mean + 2*(n_std/10)
n_stdminus = n_mean - 2*(n_std/10)

rc_mean = np.mean(r_cooplist, axis = 1)
rc_std = np.std(r_cooplist, axis = 1)
rc_stdplus = rc_mean + 2*(rc_std/10)
rc_stdminus = rc_mean - 2*(rc_std/10)

pc_mean = np.mean(p_cooplist, axis = 1)
pc_std = np.std(p_cooplist, axis = 1)
pc_stdplus = pc_mean + 2*(pc_std/10)
pc_stdminus = pc_mean - 2*(pc_std/10)

rd_mean = np.mean(r_deflist, axis = 1)
rd_std = np.std(r_deflist, axis = 1)
rd_stdplus = rd_mean + 2*(rd_std/10)
rd_stdminus = rd_mean - 2*(rd_std/10)

pd_mean = np.mean(p_deflist, axis = 1)
pd_std = np.std(p_deflist, axis = 1)
pd_stdplus = pd_mean + 2*(pd_std/10)
pd_stdminus = pd_mean - 2*(pd_std/10)

gini_mean = np.mean(gini_list, axis = 1)
gini_std = np.std(gini_list, axis = 1)
gini_stdplus = gini_mean + 2*(gini_std/10)
gini_stdminus = gini_mean - 2*(gini_std/10)

defnbrmean = np.mean(def_neighbourlist, axis = 1)
defnbrstd = np.std(def_neighbourlist, axis = 1)
defnbrstdplus = defnbrmean + 2*(defnbrstd/10)
defnbrstdminus = defnbrmean - 2*(defnbrstd/10)

coopnbrmean = np.mean(coop_neighbourlist, axis = 1)
coopnbrstd = np.std(coop_neighbourlist, axis = 1)
coopnbrstdplus = coopnbrmean + 2*(coopnbrstd/10)
coopnbrstdminus = coopnbrmean - 2*(coopnbrstd/10)

degcoopmean = np.mean(deg_coop, axis = 1)
degcoopstd = np.std(deg_coop, axis = 1)
degcoopstdplus = degcoopmean + 2*(degcoopstd/10)
degcoopstdminus = degcoopmean - 2*(degcoopstd/10)

degdefmean = np.mean(deg_def, axis = 1)
degdefstd = np.std(deg_def, axis = 1)
degdefstdplus = degdefmean + 2*(degdefstd/10)
degdefstdminus = degdefmean - 2*(degdefstd/10)

degcoopprevmean = np.mean(deg_coop_prev, axis = 1)
degcoopprevstd = np.std(deg_coop_prev, axis = 1)
degcoopprevstdplus = degcoopprevmean + 2*(degcoopprevstd/10)
degcoopprevstdminus = degcoopprevmean - 2*(degcoopprevstd/10)

degdefprevmean = np.mean(deg_def_prev, axis = 1)
degdefprevstd = np.std(deg_def_prev, axis = 1)
degdefprevstdplus = degdefprevmean + 2*(degdefprevstd/10)
degdefprevstdminus = degdefprevmean - 2*(degdefprevstd/10)

#np.savez('p0=0.82_frac=0.5.npz', gini_mean=gini_mean, gini_std=gini_std)

g_mean = np.mean(gincof)
g_std = np.std(gincof)
mu_gini = gincof*(sumwealth/50)

print("Final Gini", g_mean, "+-", 2*(g_std/10))
print("Adjusted Gini", np.mean(mu_gini), " +- ", 2*(np.std(mu_gini))/10)
      
initgini /= 100
print("init gini", initgini)
coop_mean = np.mean(coop_num)
coop_std = np.std(coop_num)
print("Fraction of cooperators", coop_mean, "+-", 2*(coop_std/10))

print("errcount", error_count)

print('coop errors', coop_errors)
print('def errors', def_errors)

print("Rich cooperators", rc_mean[-1], "+-", 2*(rc_std[-1]/10))
print("Poor cooperators", pc_mean[-1], "+-", 2*(pc_std[-1]/10))
print("Rich defectors", rd_mean[-1], "+-", 2*(rd_std[-1]/10))
print("Poor defectors", pd_mean[-1], "+-", 2*(pd_std[-1]/10))

richlist = r_cooplist + r_deflist
poorlist = p_cooplist + p_deflist
richmean = np.mean(richlist, axis = 1)
richstd = np.std(richlist, axis = 1)
poormean = np.mean(poorlist, axis = 1)
poorstd = np.std(poorlist, axis = 1)
print("Rich players", richmean[-1], "+-", 2*(richstd[-1]/10))
print("Poor players", poormean[-1], "+-", 2*(poorstd[-1]/10))


print("Rich cooperator wealthdiff", statistics.mean(rcdifflist), " +- ", 2*statistics.stdev(rcdifflist)/math.sqrt(len(rcdifflist)))
print("Poor cooperator wealthdiff", statistics.mean(pcdifflist), " +- ", 2*statistics.stdev(pcdifflist)/math.sqrt(len(pcdifflist)))
print("Rich defector wealthdiff", statistics.mean(rddifflist), " +- ", 2*statistics.stdev(rddifflist)/math.sqrt(len(rddifflist)))
print("Poor defector wealthdiff", statistics.mean(pddifflist), " +- ", 2*statistics.stdev(pddifflist)/math.sqrt(len(pddifflist)))

print("Final no of selfish environments", s_mean[-1], " + ", (s_stdplus[-1]-s_mean[-1]))

plt.figure(0)
x_axis = np.arange(1,nrounds)
#plt.xlim(0,52)
plt.ylim(0,68)
plt.plot(x_axis, c_mean, 'bo--', label='Cooperative environments')
plt.fill_between(x_axis, c_stdplus, c_stdminus, facecolor="blue", color='blue', alpha=0.2 )
plt.plot(x_axis, s_mean, 'ro--', label='Selfish environments')
plt.fill_between(x_axis, s_stdplus, s_stdminus, facecolor="orange", color='orange', alpha=0.2 )
plt.plot(x_axis, n_mean, 'go--', label='Neutral environments')
plt.fill_between(x_axis, n_stdplus, n_stdminus, facecolor="green", color='green', alpha=0.2 )
#plt.hlines(50,0,20)
plt.legend(loc="upper right")
plt.xlabel("Rounds")
plt.ylabel("Number of local strategy environments")


plt.figure(1)
x_axis = np.arange(nrounds)
plt.xlim(-2,52)
plt.ylim(0,40)
plt.plot(x_axis, rc_mean, 'bo--', label='Rich cooperators')
plt.fill_between(x_axis, rc_stdplus, rc_stdminus, facecolor="blue", color='blue', alpha=0.2 )
plt.plot(x_axis, pc_mean, 'go--', label='Poor cooperators')
plt.fill_between(x_axis, pc_stdplus, pc_stdminus, facecolor="green", color='green', alpha=0.2 )
plt.plot(x_axis, rd_mean, 'o--', color= 'orange', label='Rich defectors')
plt.fill_between(x_axis, rd_stdplus, rd_stdminus, facecolor="orange", color='orange', alpha=0.2 )
plt.plot(x_axis, pd_mean, 'ro--', label='Poor defectors')
plt.fill_between(x_axis, pd_stdplus, pd_stdminus, facecolor="magenta", color='magenta', alpha=0.2 )
plt.legend(loc="upper right")
plt.xlabel("Rounds")
plt.ylabel("Number of players")

plt.figure(2)
x_axis = np.arange((nrounds+1))
#plt.ylim(0.0,0.25)
plt.plot(x_axis, gini_mean, 'bo--')
plt.fill_between(x_axis, gini_stdplus, gini_stdminus, facecolor="blue", color='blue', alpha=0.2 )
plt.xlabel("Rounds")
plt.ylabel("Average Gini coefficient")

plt.figure(3)
#plt.ylim(0,40)
x_axis = np.arange(nrounds)
plt.plot(x_axis, defnbrmean, 'ro--', label='Defector neighbours')
plt.fill_between(x_axis, defnbrstdplus, defnbrstdminus, facecolor="magenta", color='magenta', alpha=0.2 )
plt.plot(x_axis, coopnbrmean, 'bo--', label='Cooperator neighbours')
plt.fill_between(x_axis, coopnbrstdplus, coopnbrstdminus, facecolor="blue", color='blue', alpha=0.2 )
plt.legend(loc="upper right")
plt.xlabel("Rounds")
plt.ylabel("Average number of neighbours of each type")

plt.figure(4)
plt.scatter(avg_stratlist, final_wealthlist, c = colour_list)
pcorr1 = stats.pearsonr(avg_stratlist, final_wealthlist)
scorr1 = stats.spearmanr(avg_stratlist, final_wealthlist)
plt.xlabel("Times defected")
plt.ylabel("Change in wealth (Final - Initial)")
print("Pearson's corr", pcorr1)
print("Spearman's corr", scorr1)
print("Mean number of times defected", statistics.mean(avg_stratlist))

plt.figure(5)
plt.scatter(avg_stratlist, deglist_large, c = colour_list)
pcorr2 = stats.pearsonr(avg_stratlist, deglist_large)
scorr2 = stats.spearmanr(avg_stratlist, deglist_large)
plt.xlabel("Times defected")
plt.ylabel("Final degree")
print("Pearson's corr", pcorr2)
print("Spearman's corr", scorr2)

plt.figure(6)
x_axis = np.arange(nrounds)
#plt.xlim(-2,52)
plt.ylim(20,30)
plt.plot(x_axis, richmean, 'bo--', label='Rich players')
#plt.fill_between(x_axis, (richmean+richstd), (richmean-richstd), facecolor="blue", color='blue', alpha=0.2 )
plt.plot(x_axis, poormean, 'go--', label='Poor players')
#plt.fill_between(x_axis, pc_stdplus, pc_stdminus, facecolor="green", color='green', alpha=0.2 )
plt.legend(loc="upper right")
plt.xlabel("Rounds")
plt.ylabel("Number of players")

plt.figure(7)
x_axis = np.arange(nrounds)
#plt.xlim(-2,52)
#plt.ylim(12,45)
plt.plot(x_axis, rc_mean+pc_mean, 'bo--', label='Cooperator players')
#plt.fill_between(x_axis, (richmean+richstd), (richmean-richstd), facecolor="blue", color='blue', alpha=0.2 )
plt.plot(x_axis, rd_mean+pd_mean, 'go--', label='Defector players')
#plt.fill_between(x_axis, pc_stdplus, pc_stdminus, facecolor="green", color='green', alpha=0.2 )
plt.legend(loc="upper right")
plt.xlabel("Rounds")
plt.ylabel("Number of players")


plt.figure(8)
plt.scatter(avg_stratlist, degchange_large, c = colour_list)
pcorr3 = np.corrcoef(avg_stratlist, degchange_large)
plt.xlabel("Times defected")
plt.ylabel("Change in degree")
print("Pearson's corr", pcorr3)
print("Mean number of times defected", statistics.mean(avg_stratlist)," +- ", statistics.stdev(avg_stratlist))
print("Median number of times defected", statistics.median(avg_stratlist))

plt.figure(9)
plt.ylim(11,45)
x_axis = np.arange(nrounds)
plt.plot(x_axis, degdefmean, 'ro--', label='Average degree of defectors')
plt.fill_between(x_axis, degdefstdplus, degdefstdminus, facecolor="magenta", color='magenta', alpha=0.2 )
plt.plot(x_axis, degcoopmean, 'bo--', label='Average degree of cooperators')
plt.fill_between(x_axis, degcoopstdplus, degcoopstdminus, facecolor="blue", color='blue', alpha=0.2 )
plt.legend(loc="upper right")
plt.xlabel("Rounds")
plt.ylabel("Average degree of players of each type")

plt.figure(10)
plt.ylim(11,45)
x_axis = np.arange(1,nrounds)
plt.plot(x_axis, degdefprevmean, 'ro--', label='Average degree of defectors from previous round')
plt.fill_between(x_axis, degdefprevstdplus, degdefprevstdminus, facecolor="magenta", color='magenta', alpha=0.2 )
plt.plot(x_axis, degcoopprevmean, 'bo--', label='Average degree of cooperators from previous round')
plt.fill_between(x_axis, degcoopprevstdplus, degcoopprevstdminus, facecolor="blue", color='blue', alpha=0.2 )
plt.legend(loc="upper right")
plt.xlabel("Rounds")
plt.ylabel("Average degree of players of each type")

plt.figure(11)
plt.scatter(final_stratlist_large, deglist_large, c = colour_list)
pcorr3 = np.corrcoef(final_stratlist_large, deglist_large)
plt.xlabel("Last decision")
plt.ylabel("Final degree")
print("Pearson's corr with last decision", pcorr3)

numdefect0 = []
numdefect1 = []
for i in range(5000):
    if(final_stratlist_large[i]==0):
        numdefect0.append(avg_stratlist[i])
    elif(final_stratlist_large[i]==1):
        numdefect1.append(avg_stratlist[i])
        

plt.figure(12)
plt.boxplot([numdefect0, numdefect1], whis = 1.5)
plt.xticks([1, 2], ['Cooperator', 'Defector'])
plt.xlabel("Last decision")
plt.ylabel("Number of times defected")
pcorr = np.corrcoef(final_stratlist_large, avg_stratlist)
print("Corr number of defections with last decision", pcorr)


print("Final defector degree", degdefmean[-1], " +- ", (degdefstdplus[-1] - degdefmean[-1]))
print("Final cooperator degree", degcoopmean[-1]," +- ", (degcoopstdplus[-1] - degcoopmean[-1]))

print("Final number of defector neighbours", defnbrmean[-1], " +- ", (defnbrstdplus[-1] - defnbrmean[-1]))
print("Final number of cooperator neighbours", coopnbrmean[-1], " +- ", (coopnbrstdplus[-1] - coopnbrmean[-1]))



#pcorr4 = stats.pearsonr(final_wealthlist, deglist_large)
#print("Pearson's corr 4", pcorr4)

#plt.figure(7)
#plt.plot(frequencies[0], frequencies[1], 'bo--')

plt.show()


