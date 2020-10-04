import numpy as np

def quantize(x, num_bits, alpha):
    min_q_val = -127, 
    max_q_val = 127
    scale = 127/alpha
    q = (np.around(x*scale)).astype(np.int8)
    x = q/scale
    return x

def mse_histogram_clip(bin_x, bin_y, num_bits, alpha):
   # Clipping error: sum over bins outside the clip threshold
   idx = np.abs(bin_x) > alpha
   mse = np.sum((np.abs(bin_x[idx]) - alpha)**2 * bin_y[idx])
   # Quantization error: sum over bins inside the clip threshold
   idx = np.abs(bin_x) <= alpha
   bin_xq = quantize(bin_x[idx], num_bits, alpha)
   mse += np.sum((bin_x[idx] - bin_xq)**2 * bin_y[idx])
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

    # Build histogram
    max_abs = np.max(np.abs(values))
    bin_range = (-max_abs, max_abs)
    bin_y, bin_edges = np.histogram(values, bins=1001, range=bin_range,
                                    density=True)
    bin_x = 0.5*(bin_edges[:-1] + bin_edges[1:])

    # Pick the best fitting distribution
    mse_gauss = mse_histogram_clip(bin_x, bin_y, num_bits, alpha_g)
    mse_laplace = mse_histogram_clip(bin_x, bin_y, num_bits, alpha_l)

    alpha_best = alpha_g if mse_gauss < mse_laplace else alpha_l
    print(" ACIQ alpha_best = %7.4f / %7.4f" % (alpha_best, max_abs))
    return alpha_best



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
    print('In comput loss xmin:', xmin)
    print('In comput loss xmax:', xmax)
    print('In comput loss scale:', scale)
    
    dq = qd(x,scale)
    loss = np.square(np.subtract(x, dq)).mean()
    print('In comput loss :', loss)
    return (loss)


def find_clip_greedy_search(x, bins, r):
    xmin = cur_min = np.min(x)
    xmax = cur_max = np.max(x)

    loss = compute_loss(x, xmin, xmax)
    stepsize = (xmax - xmin)/bins
    min_steps = bins * (1 - r) * stepsize
    print('max:', xmax)
    print('min:', xmin)
    print('min_steps:', min_steps)
    print('stepsize:', stepsize)


    while cur_min + min_steps < cur_max:
        print('cur_min:', cur_min)
        print('cur_max:', cur_max)
        print('loss:', loss)

        loss_l = compute_loss(x, cur_min+stepsize, cur_max)
        loss_r = compute_loss(x, cur_min, cur_max-stepsize)
        
        print('loss_l:', loss_l)
        print('loss_r:', loss_r)

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
    return xmin, xmax


    