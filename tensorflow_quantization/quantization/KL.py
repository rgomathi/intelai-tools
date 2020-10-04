import os
import sys

import numpy as np
import tensorflow as tf
from scipy.stats import entropy
import argparse


def quantize_bins(P,NUM_QBINS):
    p_dims=len(P)
    q_dims=NUM_QBINS	# will be NUM_QBINS always
        
    qfactor=np.int(np.ceil(np.float(p_dims)/q_dims))
    
    #Q=[sum(P[i:min(i+qfactor,p_dims)]) for i in ]
    Q=[]
    
    for i in range(NUM_QBINS):
        Q.append(sum(P[qfactor*i:min((i+1)*qfactor,p_dims)]))

    return Q

def dequantize_bins(Q,P):
    p_dims=len(P)
    q_dims=len(Q)	# will be NUM_QBINS always
    qfactor=np.int(np.ceil(np.float(p_dims)/q_dims))
    dequant_Q=[0]*p_dims
    qfac_new=[]
    for i in range(q_dims):
        qfac_new.append(np.count_nonzero(P[qfactor*i:min((i+1)*qfactor,p_dims)]))
    
    for j in range(p_dims):
        qfac_idx=j//qfactor
        if P[j]!=0:
            dequant_Q[j]=np.float(Q[qfac_idx])/qfac_new[qfac_idx] 
    
    #num_values=[i for i i]
    #dequant_Q=[0]*expand_dims
    return dequant_Q

def compute_entropy(p,q):
    p,q = np.asarray(p), np.asarray(q)
    p_sum,q_sum = np.sum(p),np.sum(q)
   
    p = p/p_sum
    q = q/q_sum
    ret = np.sum(np.where(p*q!=0, p * (np.log(p/q)), 0))
    return ret

def smoothDistribution(p):
    #is_zeros,is_non_zeros = len(p), len(p)
    eps = 0.0001
    is_zeros = [1 if x==0 else 0 for x in p ]
    is_non_zeros = [1 if x!=0 else 0 for x in p]
    n_zeros = sum(is_zeros)
    n_nonzeros = len(is_zeros) - n_zeros
    if n_nonzeros==0:
        return []
    
    eps1 = eps*float(n_zeros)/float(n_nonzeros)
    if eps1>=1:
        return []
    ret = p
    for i in range(len(p)):
        ret[i] = ret[i] + is_zeros[i]-eps1*is_non_zeros[i]
    return ret  

def get_pos_threshold(H,NUM_QBINS,max_val, HE):
        num_Pbins=len(H)
        res=[]
        res_edges=[]
        
        H = [float(i) for i in H]
        for i in range(NUM_QBINS,num_Pbins+1):
                ref_distribution_P=H[0:(i)]
                outliers_count=sum(H[i:])
                ref_distribution_P[i-1]=ref_distribution_P[i-1]+outliers_count
                candidate_distribution_Q=quantize_bins(H[0:(i)],NUM_QBINS)
                Q=dequantize_bins(candidate_distribution_Q,H[0:(i)])

                p = smoothDistribution(ref_distribution_P)
                q = smoothDistribution(Q)
                res.append(compute_entropy(p,q))
                res_edges.append(HE[i-1])
        
        if (len(res)==0):
            pos_idx = len(HE) - 1
            thresh_HE = HE[pos_idx]
            return(pos_idx, thresh_HE)
	
        #res[0] = 1
        min_kl=np.nanmin(res)
        pos_idx=np.argmin(res)
        
        thresh_HE = res_edges[pos_idx]
        
        #max_thresh=((pos_idx+0.5)*max_val)/num_Pbins
        
        # min_thresh=-max_thresh
        
        
        #return(max_thresh,pos_idx, thresh_HE)
        return(pos_idx, thresh_HE)


def get_neg_threshold(H,NUM_QBINS,min_val,HE):
        num_Pbins=len(H)
        res=[]
        res_edges=[]
        H = [float(i) for i in H]
        #for i in range(NUM_QBINS,num_Pbins):
        
        for i in reversed(range(0, (num_Pbins-NUM_QBINS+1))):
                ref_distribution_P=H[(i):]
                outliers_count=sum(H[0:i])
                
                ref_distribution_P[0]=ref_distribution_P[0]+outliers_count
                
                candidate_distribution_Q=quantize_bins(H[i:],NUM_QBINS)
                
                Q=dequantize_bins(candidate_distribution_Q,H[i:])
                #res.append(entropy(ref_distribution_P,Q))
                p = smoothDistribution(ref_distribution_P)
                q = smoothDistribution(Q)
                #res.append(compute_entropy(ref_distribution_P,Q))
                res.append(compute_entropy(p,q))
                res_edges.append(HE[i])
	
        if (len(res)==0):
            neg_idx = 0
            thresh_HE = HE[neg_idx]
            return(neg_idx, thresh_HE)
                                                        

        min_kl=np.nanmin(res)
        neg_idx=np.argmin(res)
        
        thresh_HE = res_edges[neg_idx]
        #min_thresh=((neg_idx-0.5)*min_val)/num_Pbins
        #return(min_thresh,neg_idx, thresh_HE)
        return(neg_idx, thresh_HE)

def get_symmetric_threshold(H_orig,NUM_QBINS,max_val,min_val,HE,HE_orig):
    H = [i for i in H_orig]    
    zero_bin=int((-min_val*len(H))/(max_val-min_val))
    
    num_Pbins=min(len(H[zero_bin:]),len(H[0:zero_bin]))
    res=[]
    res_edges=[]
    
    for i in range(NUM_QBINS//2,num_Pbins):
        ref_distribution_P=H[(zero_bin-i):(zero_bin+i)]
        outliers_count_pos=sum(H[zero_bin+i:])
        outliers_count_neg=sum(H[0:zero_bin-i])
        ref_distribution_P[-1]=ref_distribution_P[-1]+outliers_count_pos
        ref_distribution_P[0]=ref_distribution_P[0]+outliers_count_neg

        candidate_distribution_Q=quantize_bins(H[(zero_bin-i):(zero_bin+i)],NUM_QBINS)
        Q=dequantize_bins(candidate_distribution_Q,H[(zero_bin-i):(zero_bin+i)])
        
        p = [float(j) for j in ref_distribution_P]
        
        p = smoothDistribution(p)
        q = smoothDistribution(Q)
        res.append(compute_entropy(p, q))
        res_edges.append(HE[i-1])
        #res.append(entropy(ref_distribution_P,Q))
        #res[0] = 1
    if (len(res)==0):
        if (len(H[zero_bin:]) < NUM_QBINS/2):
            idx = len(HE) - 1            
            return(HE_orig[zero_bin+idx], HE_orig[zero_bin-idx], idx)
        else:
            idx = zero_bin
            return(HE_orig[zero_bin+idx], HE_orig[zero_bin-idx], idx)
          

    min_kl=np.nanmin(res)
    idx=np.argmin(res)
    thresh_HE = res_edges[idx]
    print('Actal edge at idx :', thresh_HE)
    print('Actual pos-edge at idx from full HE :', HE_orig[zero_bin+NUM_QBINS//2-1+idx])
    print('Actual neg-edge at idx from full HE :', HE_orig[zero_bin-NUM_QBINS//2-idx])
    print("Min is {} at index {}".format(min_kl,idx))

    min_thresh=((idx-0.5)*min_val)/num_Pbins
    max_thresh=-1*min_thresh
    print(max_thresh)
    print(min_thresh)
    #return(max_thresh,min_thresh,idx)
    return(HE_orig[zero_bin+NUM_QBINS//2-1+idx], HE_orig[zero_bin-NUM_QBINS//2-idx], idx)
    

def get_threshold(arr, bins, NUM_QBINS):
    hist, hist_edges = np.histogram(arr, bins)
    hist_edges_min = np.min(hist_edges)
    hist_edges_max = np.max(hist_edges)
    hist_len = len(hist)
    zero_bin = int((-hist_edges_min * hist_len) / (hist_edges_max - hist_edges_min))
    a_max, a_min, idx = get_symmetric_threshold(hist, NUM_QBINS, hist_edges_max,
                                                   hist_edges_min, hist_edges[zero_bin:], hist_edges)
    print('index :', idx)                                                   
    return (a_max, a_min)                                                   

'''

#import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--input-hist', type=str,help='input activations histogram file',dest='input',required=True)
parser.add_argument('--output-thresh', type=str,help='filename to write max and Min thresholds',dest='output',required=True)
args = parser.parse_args()

filename, file_ext = os.path.splitext(args.input)
hist_edges_filename = filename + '_edges' + file_ext

#print(args.input, filename, hist_edges_filename)

hist = np.load(args.input)
hist_edges = np.load(hist_edges_filename)
hist_shape = hist.shape
hist_edges_shape = hist_edges.shape

#print(hist_shape)
#print(hist_edges_shape)

#find max and min
#print(hist_edges[:10])

hist_edges_min = np.amin(hist_edges)
hist_edges_max = np.amax(hist_edges)
#print(hist_edges_min)
#print(hist_edges_max)

#print(np.where(hist_edges == hist_edges_min))
#print(np.where(hist_edges == hist_edges_max))

hist_min = np.amin(hist)
hist_max = np.amax(hist)
#print(hist_min)
#print(hist_max)

#print(np.where(hist == hist_min))
#print(np.where(hist == hist_max))





thresholds = []
hist_len=len(hist)
zero_bin=int((-hist_edges_min*hist_len)/(hist_edges_max-hist_edges_min))
#print('hist_len :', hist_len)
print('zero_bin :', zero_bin)
print('Sum of hist :',sum(hist))


    #call symmetric
    sym_max_thresh, sym_min_thresh, sym_idx = get_symmetric_threshold(hist, 256, hist_edges_max, hist_edges_min, hist_edges[zero_bin:], hist_edges)

    print('symmetric thgresholds : max/min/idx :', sym_max_thresh, sym_min_thresh, sym_idx)

    thresholds.append(sym_max_thresh)
    thresholds.append(sym_min_thresh)

    #thresholds.append(hist_edges[zero_bin+NUM_QBINS/2-1+idx])
    #thresholds.append(hist_edges[zero_bin-NUM_QBINS/2-idx])
    print(thresholds)
    np.save(args.output, thresholds)



#print("hist_edges[sym_idx] : ", hist_edges[sym_idx] )
#print("hist_edges[2159+128+sym_idx] : ", hist_edges[2159+128+sym_idx] ) #2159 is zero_bin, in the loop i startes with 128 i.e 256/2


# call independent

H_pos=hist[zero_bin:]
H_neg=hist[0:zero_bin]
NUM_BINS=128
pos_max_thresh,pos_idx, pos_threshold=get_pos_threshold(H_pos,NUM_BINS,hist_edges_max, hist_edges[zero_bin:])
neg_min_thresh,neg_idx, neg_threshold=get_neg_threshold(H_neg,NUM_BINS,hist_edges_min, hist_edges[0:zero_bin+1])

thresholds.append(pos_threshold)
thresholds.append(neg_threshold)
#print(hist)
print('Independent thresholds : pos_max/neg_min/pos_idx/neg_idx : ', thresholds[0], thresholds[1], pos_idx, neg_idx)

#print('hist edges corresponding to pos_idx :', hist_edges[zero_bin+NUM_BINS+pos_idx], hist_edges[zero_bin+NUM_BINS+pos_idx+1], hist_edges[zero_bin+NUM_BINS+pos_idx-1])
idx = zero_bin - NUM_BINS


#print('hist edges corresponding to neg_idx :', hist_edges[idx-neg_idx], hist_edges[idx-neg_idx+1], hist_edges[idx-neg_idx-1])

#print('hist corresponding to pos_idx :', hist[zero_bin+NUM_BINS+pos_idx], hist[zero_bin+NUM_BINS+pos_idx+1], hist[zero_bin+NUM_BINS+pos_idx-1])
#print('hist corresponding to neg_idx :', hist[idx-neg_idx], hist[idx-neg_idx+1], hist[idx-neg_idx-1])

#print('hist corresponding to hist[0], hist[1], hist[2], hist[7999], hist[8000] :', hist[0], hist[1], hist[2], hist[7999], hist[8000])

#both
#thresh=abs(max([pos_max_thresh,neg_min_thresh],key=abs))

#print ('Both : thresh : ', thresh)
#np.save('pos_neg_threshold', thresholds)
np.save(args.output, thresholds)
'''