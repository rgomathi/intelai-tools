from tensorflow.python.framework import tensor_util
import numpy as np
from tensorflow.python.framework import dtypes
import tensorflow as tf
            
def generate_const(gr, all_const, _node_infor, a_min, a_max ):
    matmul_weight_name = all_const["weight_node_name"]
    bias_name= all_const["bias_node_name"]
    weight_input_node = gr.get_node_byname(matmul_weight_name)
    weight_float_tensor = tensor_util.MakeNdarray(weight_input_node.attr["value"].tensor)
    bias_input_node = gr.get_node_byname(bias_name)
    bias_float_tensor = tensor_util.MakeNdarray(bias_input_node.attr["value"].tensor)



    #fp32_mm_wei = tf.concat([weight_float_tensor[0:457,:], weight_float_tensor[457+256+256:457+256+256+499, :]],0)
    #int8_mm_wei = tf.concat([weight_float_tensor[457:457+256, :],weight_float_tensor[457+256:457+256+256, :], weight_float_tensor[457+256+256+499:, :]], 0)

    fp32_mm_wei = np.concatenate([weight_float_tensor[0:457,:], weight_float_tensor[457+256+256:457+256+256+499, :]],0)
    int8_mm_wei = np.concatenate([weight_float_tensor[457:457+256, :],weight_float_tensor[457+256:457+256+256, :], weight_float_tensor[457+256+256+499:, :]], 0)
    print('weight_float_tensor :', weight_float_tensor)
    print('weight_float_tensor shape:', weight_float_tensor.shape)
    print('fp32_mm_wei :', fp32_mm_wei)
    print('fp32_mm_wei shape :', fp32_mm_wei.shape)

    '''
    ##### weight per channel quantization ############
    # get the max values based on axis 0 since, output channel is dim 1
    epsilon = 1e-4  # Needs to be set empirically if accuracy is not satisfactory
    
    #find the min/max from kl for each column
    print('length of weight_float_tensor column')
    print(len(weight_float_tensor[0]))

    max_list = []
    min_list = []


    ranges = np.abs(weight_float_tensor).max(axis=(0))
    min_list = np.min(weight_float_tensor, axis=0)
    max_list = np.max(weight_float_tensor, axis=0)
    print('min_list :', min_list)
    print('max_list :', max_list)
    print('ranges :', ranges)
    

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
    



    ##### update all_const #####
    #all_const["input_shape"] = input_shape    
    all_const["b"] = bias_float_tensor
    all_const["wt_for_fp32"] = fp32_mm_wei
    all_const["wt_for_int8"] = int8_mm_wei

    #all_const["w_min"] = min_list #min_value
    #all_const["w_max"] = max_list #max_value
    #all_const["bias_float"] = bias_float_tensor #max_value
    
    _node_infor["const_bias"]["const_v"] = bias_float_tensor
    _node_infor["const_wt_fp32"]["const_v"] = fp32_mm_wei
    _node_infor["const_wt_int8"]["const_v"] = int8_mm_wei
    #_node_infor["const_w_min"]["const_v"] = min_list #min_value
    #_node_infor["const_w_max"]["const_v"] = max_list #max_value
    #_node_infor["const_bz_qint32"]["const_v"] = qint32_bias
    #_node_infor["const_bias"]["const_v"] = bias_float_tensor

