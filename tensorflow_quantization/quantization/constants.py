from tensorflow.python.framework import tensor_util
import numpy as np
from tensorflow.python.framework import dtypes
import tensorflow as tf
import KL
import hist_l2norm as l2norm
import aciq as ag

def MSE(fp32, fr_min, fr_max):
    scale = 127/(np.max(np.abs([fr_min, fr_max])))
    q = (np.around(fp32*scale)).astype(np.int8)
    d = (q/scale).astype(np.float)
    mse = np.square(np.subtract(fp32, d)).mean()
    # print(fp32)
    # print(q)
    # print(d)
    return mse

#have to debug this later
def MSE_MF(fp32, fr_min, fr_max):
    print('fr_min and fr_max :', fr_min, fr_max)
    scale = 255/(fr_max-fr_min)
    q = (np.around((fp32-fr_min)*scale)).astype(np.uint8)
    d = (q/scale + fr_min).astype(np.float)
    mse = np.square(np.subtract(fp32, d)).mean()
    print(fp32)
    print(q)
    print(d)
    return mse

def generate_const(gr, all_const, _node_infor, clip_method, a_min, a_max ):

    act = np.load('../../work_dir/bd/activations.npy')
    print(act.shape)
    matmul_weight_name = all_const["weight_node_name"]
    bias_name= all_const["bias_node_name"]
    weight_input_node = gr.get_node_byname(matmul_weight_name)
    weight_float_tensor = tensor_util.MakeNdarray(weight_input_node.attr["value"].tensor)
    bias_input_node = gr.get_node_byname(bias_name)
    bias_float_tensor = tensor_util.MakeNdarray(bias_input_node.attr["value"].tensor)

    a_min = np.min(act)
    a_max = np.max(act)
    w_min = np.min(weight_float_tensor)
    w_max = np.max(weight_float_tensor)

    if (clip_method == "kl"):
        # clip activation and weight
        a_max, a_min = KL.get_threshold(act, 1001, 256)        
        # w_max, w_min = KL.get_threshold(weight_float_tensor, 1001, 256) 
        
        # result 
        # MSE of activation with a_min -1.1104698181152344 a_max 1.098185658454895 is 0.0012414596203425777
        # MSE of weight with w_min -0.3780921399593353 w_max 0.3753531873226166 is 1.2826524759306142e-05       
        # graph MSE =2.6995756e-05
    elif (clip_method == 'hist_apprx'):
        hist_apprx_act = l2norm.HistMethods(act, 1001, 256)
        a_min, a_max = hist_apprx_act.hist_approx()
        # hist_apprx_wt = l2norm.HistMethods(weight_float_tensor, 1001, 256)
        # w_min, w_max = hist_apprx_wt.hist_approx()
        
        # a_min and a_max -2.1238529682159424 3.4973917348044257 , MSE - 0.0005272489574935392
        # w_min and w_max -0.4037606082596145 0.3738432779655114 , MSE -  9.95771682321469e-06
        # MSE = 1.4419837e-05
    elif (clip_method == 'hist_brute'):
        hist_brute_act = l2norm.HistMethods(act, 201, 256)
        a_min, a_max = hist_brute_act.hist_brute()
        # hist_brute_wt = l2norm.HistMethods(weight_float_tensor, 1001, 256)
        # w_min, w_max = hist_brute_wt.hist_brute()

        # hist_brute_act = l2norm.HistMethods(act, 201, 256)
        # MSE of activation with a_min -1.4336993053777896 a_max 3.5267801462714345 is 0.0005234494730955198
        # MSE of weight with w_min -0.38693374543640746 w_max 0.349977678327418 is 1.3051032840061372e-05
    elif (clip_method == 'aciq'):
        a_max = ag.find_clip_aciq(act, 8)
        # w_max = ag.find_clip_aciq(weight_float_tensor, 8)
        a_min = -a_max
        # w_min = -w_max

    elif (clip_method == 'gs'):
        a_min, a_max = ag.find_clip_greedy_search_1(act, 200, 0.16)
        # w_min, w_max = ag.find_clip_greedy_search(weight_float_tensor, 200, 0.16)

    # a_min = -1.64317
    # a_max = 1.64317
    # a_max = 4.51301
    #a_max = 2.51301
    a_min = -1.907412185668947
    a_max =  5.93898794174193

    # w_min = -0.5391701790724762
    # w_max = 0.5391701790724762

    

    print('clip_method used :', clip_method)
    print ('a_min and a_max', a_min, a_max)
    print ('w_min and w_max', w_min, w_max)
    
    #print('MSE of activation with a_min {} a_max {} is {}'.format(a_min, a_max, MSE(act, a_min, a_max)))
    print('MSE of activation with a_min {} a_max {} is {}'.format(a_min, a_max, MSE_MF(act, a_min, a_max)))
    print('MSE of weight with w_min {} w_max {} is {}'.format(w_min, w_max, MSE(weight_float_tensor, w_min, w_max)))
    
    
 
    abs_max_act = np.max(np.abs([a_min, a_max]))
    abs_max_wt  = np.max(np.abs([w_min, w_max]))
    weight_qint8_tensor = (np.around(weight_float_tensor*127/abs_max_wt)).astype(np.int8)
    
 

    ###### Bias compensation ###########
    # compensated with B's32 = Q'a * Qw * Bf32 + Q'a * Min(Af32) * 1 *ws8.
    QaAmin  = 255 * a_min / (a_max - a_min);

    int32_bias = []

    bias_scale = 255.0 * 127.0 / ((a_max - a_min) * abs_max_wt)
    # bias_scale = 127.0 * 127.0 / (abs_max_act * abs_max_wt)
    ws8_1 = np.sum(np.array(weight_qint8_tensor, dtype=np.int32),axis=0,dtype=np.int32)
    qint32_bias = np.around((bias_float_tensor * bias_scale) + (ws8_1*QaAmin))
    # qint32_bias = np.around(bias_float_tensor * bias_scale)

  


    ##### update all_const #####
    #all_const["input_shape"] = input_shape    
    #all_const["b"] = bias_float_tensor
    all_const["a_min"] = a_min
    all_const["a_max"] = a_max
    #all_const["wt_for_fp32"] = fp32_mm_wei
    #all_const["wt_for_int8"] = int8_mm_wei
    all_const["wt_for_int8"] = weight_qint8_tensor
    all_const["w_min"] = w_min
    all_const["w_max"] = w_max
    all_const["b_comp"] = qint32_bias
    
    _node_infor["const_a_min"]["const_v"] = a_min
    _node_infor["const_a_max"]["const_v"] = a_max
    #_node_infor["const_bias"]["const_v"] = bias_float_tensor
    #_node_infor["const_wt_fp32"]["const_v"] = fp32_mm_wei
    #_node_infor["const_wt_int8"]["const_v"] = int8_mm_wei
    _node_infor["const_wt_int8"]["const_v"] = weight_qint8_tensor
    _node_infor["const_wt_min"]["const_v"] = w_min
    _node_infor["const_wt_max"]["const_v"] = w_max
    _node_infor["const_b_comp"]["const_v"] = qint32_bias#np.zeros((bias_float_tensor.shape))


    #_node_infor["const_bz_qint32"]["const_v"] = qint32_bias
    #_node_infor["const_bias"]["const_v"] = bias_float_tensor

