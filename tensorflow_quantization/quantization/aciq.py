import numpy as np
import tensorflow as tf
# from tensorflow.python.ops.gen_math_ops import quantize, dequantize

def quantize(x, alpha):
    min_q_val = -127, 
    max_q_val = 127
    scale = 127/alpha
    q = (np.around(x*scale)).astype(np.int8)
    x = q/scale
    return x

def mse_histogram_clip(bin_x, bin_y, alpha):
   # Clipping error: sum over bins outside the clip threshold
   idx = np.abs(bin_x) > alpha
#    print('alpha ', alpha)
#    print('idx ', idx)
#    print('bin_x[idx] ', bin_x[idx])
   mse = np.sum((np.abs(bin_x[idx]) - alpha)**2 * bin_y[idx])
#    print('(np.abs(bin_x[idx]) - alpha)**2 ', (np.abs(bin_x[idx]) - alpha)**2)
#    print('(np.abs(bin_x[idx]) - alpha)**2 * bin_y[idx] ', (np.abs(bin_x[idx]) - alpha)**2 * bin_y[idx])
#    print('mse ', mse)
   # Quantization error: sum over bins inside the clip threshold
   idx = np.abs(bin_x) <= alpha
   bin_xq = quantize(bin_x[idx], alpha)
   mse += np.sum((bin_x[idx] - bin_xq)**2 * bin_y[idx])
#    print('(bin_x[idx] - bin_xq)**2 * bin_y[idx] ', (bin_x[idx] - bin_xq)**2 * bin_y[idx])
#    print('mse ', mse)
   return mse

#-------------------------------------------------------------------------
# ACIQ method
#   ACIQ: Analytical Clipping for Integer Quantization of Neural Networks
#   https://arxiv.org/pdf/1810.05723.pdf
# Code taken and modified from:
#   https://github.com/submission2019/AnalyticalScaleForIntegerQuantization/blob/master/mse_analysis.py
#-------------------------------------------------------------------------
# 1. Find Gaussian and Laplacian clip thresholds
# 2. Estimate the MSE and choose the correct distribution
alpha_gauss   = {2:1.47818312, 3:1.80489289, 4:2.19227856, 5:2.57733584, 6:2.94451183, 7:3.29076248, 8:3.61691335}
alpha_laplace = {2:2.33152939, 3:3.04528770, 4:4.00378631, 5:5.08252088, 6:6.23211675, 7:7.42700429, 8:8.65265030}
gaussian_const = (0.5 * 0.35) * (1 + (np.pi * np.log(4)) ** 0.5)

def find_clip_aciq(values, num_bits):
    # Gaussian clip
    # This is how the ACIQ code calculates sigma
    sigma = ((np.max(values) - np.min(values)) * gaussian_const) / ((2 * np.log(values.size)) ** 0.5)
    #sigma = np.sqrt(np.sum((values - np.mean(values))**2) / (values.size-1))
    alpha_g = alpha_gauss[num_bits] * sigma
    # Laplacian clip
    b = np.mean(np.abs(values - np.mean(values)))
    alpha_l = alpha_laplace[num_bits] * b
    print('sigma ', sigma)
    print('alpha_g ', alpha_g)
    print('b ', b)
    print('alpha_l', alpha_l)

    # Build histogram
    max_abs = np.max(np.abs(values))
    bin_range = (-max_abs, max_abs)
    bin_range = (np.min(values), np.max(values))
    print('bin_range ', bin_range)
    bin_y, bin_edges = np.histogram(values, bins=101, range=bin_range,
                                    density=True)
    bin_x = 0.5*(bin_edges[:-1] + bin_edges[1:])
    print('bin_edges ', bin_edges)
    print('bin_x ', bin_x)
    print('bin_y ', bin_y)

    # Pick the best fitting distribution
    mse_gauss = mse_histogram_clip(bin_x, bin_y,  alpha_g)
    mse_laplace = mse_histogram_clip(bin_x, bin_y,  alpha_l)
    print('mse_gauss', mse_gauss)
    print('mse_laplace ', mse_laplace)

    alpha_best = alpha_g if mse_gauss < mse_laplace else alpha_l
    print(" ACIQ alpha_best = %7.4f / %7.4f" % (alpha_best, max_abs))
    return alpha_best


def find_clip_mmse(values, num_bits):
    # Build histogram
    # max_abs = np.max(np.abs(values))
    bin_y, bin_edges = np.histogram(values, bins=1001, density=True)
    # print('bin_edges ', bin_edges)
    # print('bin_y ', bin_y)
    # print('np.sum(bin_y) ', np.sum(bin_y))
    bin_x = 0.5*(bin_edges[:-1] + bin_edges[1:])
    bin_y /= np.sum(bin_y)
    
    # print('bin_x ', bin_x)
    # print('bin_y ', bin_y)
    # split positive and negative seperately
    hist_edges_min = np.min(bin_edges)
    hist_edges_max = np.max(bin_edges)
    hist_len = len(bin_y)
    zero_bin = int((-hist_edges_min * hist_len) / (hist_edges_max - hist_edges_min))
    print('hist_len ', hist_len)
    print('zero_bin', zero_bin)
    

    alphas = np.arange(0.01, 1, 0.01) * hist_edges_max
    print('len of alphas ', len(alphas))
    # print('alphas ', alphas)
    mses_pos = [ mse_histogram_clip(bin_x[zero_bin:], bin_y[zero_bin:], alpha)
                 for alpha in alphas ]
    alpha_best_max = alphas[np.argmin(mses_pos)]

    # alphas = np.arange(0.01, 1, 0.01)
    # alphas = alphas[::-1]*-hist_edges_min
    alphas = np.arange(0.01, 1, 0.01) * -hist_edges_min
    # print('alphas ', alphas)
    # bin_x_r = np.array(tuple(reversed(bin_x[:zero_bin])))
    # bin_y_r = np.array(tuple(reversed(bin_y[:zero_bin])))
    bin_x_r = np.flip(bin_x[:zero_bin])
    bin_y_r = np.flip(bin_y[:zero_bin])
   
    # print('bin_x_r ', bin_x_r)
    # print('bin_y_r ', bin_y_r)
    mses_neg = [ mse_histogram_clip(bin_x_r, bin_y_r, alpha)
                 for alpha in alphas ]
    print('mses_pos ', len(mses_pos))
    print('mses_neg ', len(mses_neg))
    alpha_best_min = alphas[np.argmin(mses_neg)]

    print(" alpha_best_max and alpha_best_min are",alpha_best_max, -alpha_best_min)
    return -alpha_best_min, alpha_best_max



# Greedy search based on https://arxiv.org/pdf/1911.02079.pdf

def qd(x, scale):
    # min_q_val = -127, 
    # max_q_val = 127
    # scale = 127/alpha
    q = (np.around(x*scale)).astype(np.int8)
    dq = q/scale
    return dq

def compute_loss(x, xmin, xmax):
    scale = 127/np.max(np.abs([xmin, xmax]))
    # print('In comput loss xmin:', xmin)
    # print('In comput loss xmax:', xmax)
    # print('In comput loss scale:', scale)
    
    dq = qd(x,scale)
    loss = np.square(np.subtract(x, dq)).mean()
    # print('In comput loss :', loss)
    return (loss)

def qd_mf(x, scale, xmin):
    q = (np.around((x-xmin)*scale)).astype(np.uint8)
    dq = (q/scale + xmin).astype(np.float)
    
    return dq

def compute_loss_mf(x, xmin, xmax):
    scale = 255/(xmax-xmin)
    # print('In comput loss xmin:', xmin)
    # print('In comput loss xmax:', xmax)
    # print('In comput loss scale:', scale)
    
    dq = qd_mf(x, scale, xmin)
    loss = np.square(np.subtract(x, dq)).mean()
    # print('In comput loss :', loss)
    return (loss)

    
def compute_loss_tf(x, xmin, xmax):
    # scale = 255/(xmax-xmin)
    # print('In comput loss xmin:', xmin)
    # print('In comput loss xmax:', xmax)
    # print('In comput loss scale:', scale)
    
    # dq = qd_mf(x, scale, xmin)
    # loss = np.square(np.subtract(x, dq)).mean()
    # print('In comput loss :', loss)
     
    graph =tf.Graph()
    with graph.as_default():
        q_tf, min_tf, max_tf = tf.compat.v1.quantize(x, xmin, xmax, tf.quint8,  mode='MIN_FIRST')
        dq_tf = tf.compat.v1.dequantize(q_tf, min_tf, max_tf, mode='MIN_FIRST')

        with tf.compat.v1.Session() as sess:
            dq = sess.run(dq_tf)
            loss = np.square(np.subtract(x, dq)).mean()


    return (loss)


def find_clip_greedy_search(x, bins, r):
    xmin = cur_min = np.min(x)
    xmax = cur_max = np.max(x)

    loss = compute_loss_mf(x, xmin, xmax)
    
    stepsize = (xmax - xmin)/bins
    min_steps = bins * (1 - r) * stepsize
    print('max:', xmax)
    print('min:', xmin)
    print('min_steps:', min_steps)
    print('stepsize:', stepsize)

    i = 0
    # while cur_min + min_steps < cur_max:
    while cur_min < cur_max:
        i+=1
        # print('cur_min + min_steps :', cur_min + min_steps)
        # print('cur_max :', cur_max)
        print('cur_min:', cur_min)
        print('cur_max:', cur_max)
        # print('loss:', loss)

        loss_l = compute_loss_mf(x, cur_min+stepsize, cur_max)
        loss_r = compute_loss_mf(x, cur_min, cur_max-stepsize)
        
        # print('loss_l:', loss_l)
        # print('loss_r:', loss_r)

        if loss_l < loss_r:
            cur_min = cur_min + stepsize

            if loss_l < loss:
                loss = loss_l
                xmin = cur_min
        else:
            cur_max = cur_max - stepsize
            if loss_r < loss:
                loss = loss_r
                xmax = cur_max

    print('Final max:', xmax)
    print('Final min:', xmin)
    print('loop runs for:', i)
    return xmin, xmax


def find_clip_greedy_search_1(x, bins, r):
    xmin = cur_min = np.min(x)
    xmax = cur_max = np.max(x)

    loss = compute_loss_mf(x, xmin, xmax)
    print('loss from tf', loss)
    stepsize = (xmax - xmin)/bins
    min_steps = bins * (1 - r) * stepsize
    print('max:', xmax)
    print('min:', xmin)
    # print('min_steps:', min_steps)
    print('stepsize:', stepsize)

    i=0
    # while cur_min < 0:
    #     cur_max = xmax
    #     while cur_max > 0:
    #         i+=1
    #         loss_new = compute_loss_mf(x, cur_min, cur_max)
    #         # print('cur_min:', cur_min)
    #         # print('cur_max:', cur_max)        
    #         # print('new_loss:', loss_new)
    #         if loss_new < loss:
    #             loss = loss_new
    #             xmin_new = cur_min
    #             xmax_new = cur_max    
    #             print('xmax:', xmax_new)
    #             print('xmin:', xmin_new)
    #             print('new_loss:', loss_new)

    #         cur_max = cur_max - stepsize
    #     print('cur_min:', cur_min)
    #     cur_min = cur_min + stepsize

    cur_min = -stepsize
    while cur_min > xmin:
        cur_max = stepsize
        while cur_max < xmax:
            i+=1
            loss_new = compute_loss_tf(x, cur_min, cur_max)
            # print('cur_min:', cur_min)
            # print('cur_max:', cur_max)        
            # print('new_loss:', loss_new)
            if loss_new < loss:
                loss = loss_new
                xmin_new = cur_min
                xmax_new = cur_max    
                print('xmax:', xmax_new)
                print('xmin:', xmin_new)
                print('new_loss:', loss_new)

            cur_max = cur_max + stepsize
        print('cur_min:', cur_min)
        cur_min = cur_min - stepsize

  
    print('Final max:', xmax_new)
    print('Final min:', xmin_new)
    print('loop runs for:', i)
    return xmin_new, xmax_new