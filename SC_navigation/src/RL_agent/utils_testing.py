import os
import numpy as np
from collections import namedtuple
import random
import statistics 

def show_results(results_dict,modules,repeats,file_name,mode,verbose = True):
    mean_steps,mean_score,alphas,n_goals,n_colls = read_results(results_dict,modules)  

    if verbose:
        #print('\n')
        print('Results:')
        #print('-'*30)
        print(' - Mean score = ',mean_score)
        print(' - Mean steps = ',mean_steps)

    file_name.write('\nMODE =%s\r\n'%mode)
    file_name.write(' - Mean score = %f\r\n'%mean_score)
    file_name.write(' - Mean steps = %f\r\n'%mean_steps)

    for i in range(3):
        m = modules[i]
        file_name.write(' - %s control:\r'%m)
        file_name.write('    + Occurrence = %f\r'%round(alphas[i,0],2))
        file_name.write('    + Mean = %f\r'%round(alphas[i,1],2))
        file_name.write('    + Variance = %f\r\n'%round(alphas[i,2],2))

        if verbose: 
            print(f' - {m} control:')
            print('    + Occurrence = ',round(alphas[i,0],2) )
            print('    + Mean = ',round(alphas[i,1],2))
            print('    + Variance = ',round(alphas[i,2],2))

    file_name.write(' - goals = %f\r\n'%(n_goals/repeats))
    file_name.write(' - colls = %f\r\n'%(n_colls/repeats))

    if verbose: 
        print(' - goals = ',round((n_goals/repeats*100),3), '%')
        print(' - colls = ',round((n_colls/repeats*100),3), '%')
    
    return n_goals/repeats*100,mean_score
 
def read_results(results,modules):
    n = len(results.keys())
    tot_steps = 0
    tot_colls = 0
    tot_goals = 0
    tot_scores = 0 
    alpha1 =  {'occurrence':[], 'mean': [], 'var':[]}
    alpha2 =  {'occurrence':[], 'mean': [], 'var':[]}
    alpha3 =  {'occurrence':[], 'mean': [], 'var':[]}
    alphas = np.array([alpha1,alpha2,alpha3])

    for k in results.keys():
        r = results[k]
        tot_steps += r.n_steps
        
        if r.ending== 'goal':
            tot_goals +=1
            tot_scores += r.score
        elif r.ending== 'coll':
            tot_colls += 1

        stats = get_alpha_stats(r.alpha_data)
        if stats==None:
            continue
        for m,a in zip(modules,alphas):
            a['occurrence'].append(stats[m].occorrence/r.n_steps)  
            a['mean'].append(stats[m].mean)
            a['var'].append(stats[m].variance)
      
    mean_steps = tot_steps/n
    mean_score = tot_scores/tot_goals
    mean_alpha_res = np.zeros([3,3])
    for i in range(3):
        a = alphas[i]
        mean_alpha_res[i,0] = statistics.mean(a['occurrence'])  
        mean_alpha_res[i,1] = statistics.mean(a['mean'])
        mean_alpha_res[i,2] = statistics.mean(a['var'])

    return mean_steps,mean_score,mean_alpha_res,tot_goals,tot_colls


def get_alpha_stats(data):
    Stat = namedtuple('Stat',field_names=['occorrence','mean','variance'])
    stats = {}
    data = np.array(data)
    n = data.shape[0]
    modules = ['usr','ca_r','ca_t']
    try:
        for i in range(3):
            d_i = data[:,i].tolist()
            occ_i = n - d_i.count(0.0)
            mean_i = statistics.mean(d_i)
            var_i = statistics.variance(d_i)
            stats[modules[i]] = Stat(occ_i,mean_i,var_i)
        return stats
    except:
        return None
