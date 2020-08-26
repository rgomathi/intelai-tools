from tensorflow.python.framework import tensor_util
import numpy as np
from tensorflow.python.framework import dtypes
import tensorflow as tf
from KL import get_pos_threshold
from KL import get_neg_threshold
from KL import get_symmetric_threshold

from kl_divergence import KL_Divergence

def q_m_qu_date(s_data, fix_max_i, Trange, sess=None, to_run=True):
    result =tf.user_ops.quant14bits_out(s_data, fix_max_i, t_range=Trange)
    if to_run:
        sess.run(result)
    z_max = result[0]
    z_min =  result[1]
    q_scale = result[2]
    q_h =  result[3]
    q_l =  result[4]
    return q_scale, q_h, q_l, z_max, z_min

def r_a_const(ws, WH, WL, Wq, a_Fix_max, a_T_Range, w_Trange):
    with tf.Session() as sess:
            XH_size = (1, WH.shape[0])
            C_One = tf.ones(XH_size, dtype=tf.int32)
            Wq_int32 = tf.cast(Wq, tf.int32)
            C_One_X_W= tf.matmul(C_One,Wq_int32)
            C_One_X_W_fp32=tf.cast(C_One_X_W, tf.float32)
            a_t = C_One_X_W_fp32 / ws 
            sess.run(a_t)
            a_const = a_t.eval()
            #print(a_const)
            #print(T_Range, ws , Fix_max)
            xh_wh_r =   (float(a_T_Range) /ws * float(w_Trange) /float(a_Fix_max))
            xh_wl_r = (float(a_T_Range) /ws /float(a_Fix_max))
            xl_wh_r = (float(w_Trange) /ws /float(a_Fix_max))
            xl_wl_r =  1.0 /ws /float(a_Fix_max)
            return a_const, xh_wh_r.eval(), xh_wl_r.eval(), xl_wh_r.eval(), xl_wl_r.eval()
            
def generate_const(gr, all_const, _node_infor, a_min, a_max ):
    matmul_weight_name = all_const["weight_node_name"]
    bias_name= all_const["bias_node_name"]
    weight_input_node = gr.get_node_byname(matmul_weight_name)
    weight_float_tensor = tensor_util.MakeNdarray(weight_input_node.attr["value"].tensor)
    bias_input_node = gr.get_node_byname(bias_name)
    bias_float_tensor = tensor_util.MakeNdarray(bias_input_node.attr["value"].tensor)

    ##### weight per channel quantization ############
    # get the max values based on axis 0 since, output channel is dim 1
    epsilon = 1e-4  # Needs to be set empirically if accuracy is not satisfactory
    
    #find the min/max from kl for each column
    print('length of weight_float_tensor column')
    print(len(weight_float_tensor[0]))

    max_list = []
    min_list = []
    _kl = KL_Divergence()

    #### use kl-div from ilit to fix min/max of weights
    for i in range(len(weight_float_tensor[0])):
        hist, hist_edges = np.histogram(weight_float_tensor[:, i], bins=8001)
        
        th = _kl.get_threshold(hist, 
                            hist_edges, 
                            np.min(weight_float_tensor[:,i]),
                            np.max(weight_float_tensor[:,i]),
                            num_bins=8001,
                            quantized_type=np.int8,
                            num_quantized_bins=255)

        max_list.append(th)
        min_list.append(-th)
        print('column, actual min/max and kl fixed min/max :', i, np.min(weight_float_tensor[:,i]), np.max(weight_float_tensor[:,i]), -th, th)

    np.save('min_list', min_list)
    np.save('max_list', max_list)
     
    min_list = np.load('min_list.npy')
    max_list = np.load('max_list.npy')
    
    

    abs_max = np.abs(max_list)
    abs_min = np.abs(min_list)
    ranges = np.maximum(abs_max, abs_min)
    print('min_list :', min_list)
    print('max_list :', max_list)
    print('ranges :', ranges)


    '''
    ######### use kl-div to fix min/max
    for i in range(len(weight_float_tensor[0])):
        hist, hist_edges = np.histogram(weight_float_tensor[:, i], bins=1000)
        #print(len(hist))
        #print(len(hist_edges))
        hist_edges_min = np.amin(hist_edges)
        hist_edges_max = np.amax(hist_edges)
        hist_len=len(hist)
        zero_bin=int((-hist_edges_min*hist_len)/(hist_edges_max-hist_edges_min))
        #print('zero_bin :', zero_bin)
        #print('Sum of hist :',sum(hist))

        H_pos=hist[zero_bin:]
        H_neg=hist[0:zero_bin]
        NUM_BINS=128
        #NUM_BINS=127
        #pos_idx, pos_threshold=get_pos_threshold(H_pos,NUM_BINS,hist_edges_max, hist_edges[zero_bin:])
        #neg_idx, neg_threshold=get_neg_threshold(H_neg,NUM_BINS,hist_edges_min, hist_edges[0:zero_bin+1])

        #call symmetric
        sym_max_thresh, sym_min_thresh, sym_idx = get_symmetric_threshold(hist, 255, hist_edges_max, hist_edges_min, hist_edges[zero_bin:], hist_edges)

        print('symmetric thgresholds : max/min/idx :', sym_max_thresh, sym_min_thresh, sym_idx)

        #thresholds.append(sym_max_thresh)
        #thresholds.append(sym_min_thresh)
        pos_threshold = sym_max_thresh
        neg_threshold = sym_min_thresh

        max_list.append(pos_threshold)
        min_list.append(neg_threshold)
        #print('loop :', i)
        print('column, actual min/max and kl fixed min/max :', i, np.min(weight_float_tensor[:,i]), np.max(weight_float_tensor[:,i]), neg_threshold, pos_threshold)

    np.save('min_list', min_list)
    np.save('max_list', max_list)
     
    min_list = np.load('min_list.npy')
    max_list = np.load('max_list.npy')
    
    

    abs_max = np.abs(max_list)
    abs_min = np.abs(min_list)
    ranges = np.maximum(abs_max, abs_min)
    print('min_list :', min_list)
    print('max_list :', max_list)
    print('ranges :', ranges)
    '''    

    '''
    ranges = np.abs(weight_float_tensor).max(axis=(0))
    min_list = np.min(weight_float_tensor, axis=0)
    max_list = np.max(weight_float_tensor, axis=0)
    print('min_list :', min_list)
    print('max_list :', max_list)
    print('ranges :', ranges)
    '''

    # nudging min-max values outside epsilon radius around zero
    #ranges[ranges < epsilon] = epsilon
    #min_list[np.abs(min_value) < epsilon] = -epsilon
    #max_list[np.abs(max_value) < epsilon] = epsilon
    weight_qint8_tensor = (weight_float_tensor * 127.0 / ranges).astype(np.int8)
    #print('sizeof weight_qint8_tensor', weight_qint8_tensor.shape)
    print('weight_float_tensor :')
    print(weight_float_tensor)
    #print(weight_qint8_tensor)
    #print(weight_float_tensor * 127.0 / ranges)
    #print((np.around(weight_float_tensor * 127.0 / ranges)).astype(np.int8))
    weight_qint8_tensor = (np.around(weight_float_tensor * 127.0 / ranges)).astype(np.int8)
    print('weight_qint8_tensor :')
    print(weight_qint8_tensor)
    #print('Print after epsilon check')
    #print(ranges)
    #print(min_list)
    #print(max_list)




    ###### Bias compensation ###########
    # compensated with B's32 = Q'a * Qw * Bf32 + Q'a * Min(Af32) * 1 *ws8.
    QaAmin  = 255 * a_min / (a_max - a_min);

    int32_bias = []

    bias_scale = 255.0 * 127.0 / ((a_max - a_min) * ranges)
    ws8_1 = np.sum(np.array(weight_qint8_tensor, dtype=np.int32),axis=0,dtype=np.int32)
    qint32_bias = np.around((bias_float_tensor * bias_scale) + (ws8_1*QaAmin))
    #qint32_bias = np.around((ws8_1*QaAmin))


    #print(bias_scale)
    #print(ws8_1)
    print(qint32_bias)
    '''
    for bias_index, value in enumerate(
            np.sum(np.array(weight_qint8_tensor, dtype=np.int32),
                                   axis=0,
                                   dtype=np.int32)):
            bias_scale = 255.0 * 127.0 / (
                        (a_max - a_min) *
                        max(abs(max_value[bias_index]), abs(min_value[bias_index])))
u           int32_bias.append(
                            (bias_float_tensor[bias_index] * bias_scale +
                                value * QaAmin).astype(np.int32))
    '''
#    int32_bias_t = tf.convert_to_tensor(int32_bias, tf.qint32)
    #print(int32_bias)
    
    #input_shape = (1, 1)
    
    #all_const["input_shape"] = input_shape    
    all_const["b"] = qint32_bias
    all_const["w"] = weight_qint8_tensor
    all_const["w_min"] = min_list #min_value
    all_const["w_max"] = max_list #max_value
    #all_const["bias_float"] = bias_float_tensor #max_value
    
    _node_infor["const_w"]["const_v"] = weight_qint8_tensor
    _node_infor["const_w_min"]["const_v"] = min_list #min_value
    _node_infor["const_w_max"]["const_v"] = max_list #max_value
    _node_infor["const_bz_qint32"]["const_v"] = qint32_bias
    #_node_infor["const_bias"]["const_v"] = bias_float_tensor

