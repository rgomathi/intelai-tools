from tensorflow.python.framework import dtypes
import numpy as np
from constants import generate_const

def get_nodes_control_pannel(gr, fc_cfg): 
    all_const = {
                                    "weight_node_name":fc_cfg["weight_node_name"],
                                    "bias_node_name":fc_cfg["bias_node_name"],
                                    "input_shape": None,
                                    "b": None,
                                    "w_for_fp32": None,
                                    "w_for_int8": None,
                                    "w_comp": None,
                                    #"w_max": None,
                                    #"bias_float": None,
                                }

    _node_infor = {
                            "split":
                            {
                                "name"  :   "split",
                                "op"    :   "SplitV",
                                "input_name_extra"  : False,
                                "input" :  [
                                            fc_cfg["input_node_name"],
                                            "const_size_splits",
                                            "const_split_dim"
                                           ], 
                                "attr"  :   {
                                            "T":{"type":"dtype", "v":dtypes.float32},
                                            "Tlen":{"type":"dtype", "v":dtypes.int32},
                                            "num_split":{"type":"int", "v":5}
                                            }
                            },
                            "const_size_splits":
                            {
                                "name"  : "const_size_splits",
                                "op"    : "Const",
                                "type"  : dtypes.int32,
                                "shape" : None,
                                "const_v"   :   [457, 256,256, 499, 128]
                            },
                            "const_split_dim":
                            {
                                "name"  : "const_split_dim",
                                "op"    : "Const",
                                "type"  : dtypes.int32,
                                "shape" : None,
                                "const_v"   :   1
                            },
                            "concat_1":
                            {
                                "name"  :   "concat_1",
                                "op"    :   "ConcatV2",
                                "input_name_extra"  : False,
                                "input" :  [
                                            "split",
                                            "split:3",
                                            "const_concat1_axis"
                                           ], 
                                "attr"  :   {
                                            "N":{"type":"int", "v":2},
                                            "T":{"type":"dtype", "v":dtypes.float32},
                                            "Tidx":{"type":"dtype", "v":dtypes.int32}
                                            }

                            },
                            "const_concat1_axis":
                            {
                                "name"  : "const_concat1_axis",
                                "op"    : "Const",
                                "type"  : dtypes.int32,
                                "shape" : None,
                                "const_v"   :   1
                            },
                            "concat_2":
                            {
                                "name"  :   "concat_2",
                                "op"    :   "ConcatV2",
                                "input_name_extra"  : False,
                                "input" :  [
                                            "split:1",
                                            "split:2",
                                            "split:4",
                                            "const_concat2_axis"
                                           ], 
                                "attr"  :   {
                                            "N":{"type":"int", "v":3},
                                            "T":{"type":"dtype", "v":dtypes.float32},
                                            "Tidx":{"type":"dtype", "v":dtypes.int32}
                                            }
                            },
                            "const_concat2_axis":
                            {
                                "name"  : "const_concat2_axis",
                                "op"    : "Const",
                                "shape" : None,
                                "type"  : dtypes.int32,
                                "const_v"   :   1
                            },
                            "matmul_fp32":
                            {
                                "name"  :   "matmul_fp32",
                                "op"    :   "MatMul",
                                "input_name_extra"  : False,
                                "input" :  [
                                            "concat_1",
                                            "const_wt_fp32",
                                           ], 
                                "attr"  :   {
                                            "T":{"type":"dtype", "v":dtypes.float32},
                                            "transpose_a":{"type":"bool", "v":False},
                                            "transpose_b":{"type":"bool", "v":False}
                                            }
                            },
                            "const_wt_fp32":
                            {
                                "name"  : "const_wt_fp32",
                                "op"    : "Const",
                                "type"  : dtypes.float32,
                                "shape" : None,
                                "const_v"   :   all_const["w_for_fp32"]
                            },
                            "matmul_int8":
                            {
                                "name"  :   "matmul_int8",
                                "op"    :   "MatMul",
                                "input_name_extra"  : False,
                                "input" :  [
                                            "concat_2",
                                            "const_wt_int8",
                                           ], 
                                "attr"  :   {
                                            "T":{"type":"dtype", "v":dtypes.float32},
                                            "transpose_a":{"type":"bool", "v":False},
                                            "transpose_b":{"type":"bool", "v":False}
                                            }
                            },
                            "const_wt_int8":
                            {
                                "name"  : "const_wt_int8",
                                "op"    : "Const",
                                "type"  : dtypes.float32,
                                "shape" : None,
                                "const_v"   :   all_const["w_for_int8"]
                            },
                            "addn":
                            {
                                "name"  : "addn",
                                "op"    : "AddN",
                                "input" : ["matmul_fp32", "matmul_int8"],
                                "attr"  :   {
                                            "N":{"type":"int", "v":2},
                                            "T":{"type":"dtype", "v":dtypes.float32}
                                            }
                            },
                            "bias_add":
                            {
                                "name"  : "bias_add",
                                "op"    : "BiasAdd",
                                "input" : ["addn", "const_bias"],
                                "attr"  :   {
                                            "T":{"type":"dtype", "v":dtypes.float32}
                                            }
                            },
                            "const_bias":
                            {
                                "name"  : "const_bias",
                                "op"    : "Const",
                                "type"  : dtypes.float32,
                                "shape" : None,
                                "const_v"   :   all_const["b"]
                            },
                            "relu":
                            {
                                "name"  : fc_cfg["fc_output_node_name"],
                                "op"    : "Relu",
                                "input" : ["bias_add"],
                                "attr"  :   {
                                            "T":{"type":"dtype", "v":dtypes.float32}
                                            }
                                
                            }
    
                  }

    _node_control_pannel = {
                    "Settings":{
                                "Data_Folder":"/media/data/ml/WND_SW_ROOT/tf_proxy_model_quant/models/src/"
                        },
                    "name_extra":fc_cfg["name_extra"],
                    "name_extra_connection":fc_cfg["name_extra_connection"],
                     "to_be_removed":fc_cfg["to_be_removed"],
                    "to_be_added":[_node_infor["split"],
                                   _node_infor["const_size_splits"],
                                   _node_infor["const_split_dim"],
                                   _node_infor["concat_1"],
                                   _node_infor["const_concat1_axis"],        
                                   _node_infor["concat_2"],
                                   _node_infor["const_concat2_axis"],
                                   _node_infor["matmul_fp32"],
                                   _node_infor["const_wt_fp32"], 
                                   _node_infor["matmul_int8"],        
                                   _node_infor["const_wt_int8"],
                                   _node_infor["addn"],
                                   _node_infor["bias_add"],
                                   _node_infor["const_bias"],
                                   _node_infor["relu"],
                                  ],
                    "to_be_updated":{
                                            }
                }
               
    generate_const(gr, all_const, _node_infor, fc_cfg["range_preset_min"], fc_cfg["range_preset_max"])
    #if 'PRESET' in str(fc_cfg["range_mode"]):
        #print("===> Optimizing for PRESET mode. <===")
        #nv = _node_infor["const_a"]["const_v"] * fc_cfg["range_preset_min"]
        #nvb = all_const["b"] + nv[0]
        #_node_infor["const_bias"]["const_v"] = nvb
        #_node_infor["bias_add"]["input"]= ["Add_xw","const_bias"]
        #_node_control_pannel[ "to_be_added"].remove(_node_infor[ "Mul_a_vmin"])
        #_node_control_pannel[ "to_be_added"].remove(_node_infor[ "Add_a_c"])
        #_node_control_pannel[ "to_be_added"].remove(_node_infor[ "const_a"])
    '''
        "const_bias":{ "name": "const_bias",
                          "op": "Const",
                                                "attr": None,
                                                "type": dtypes.float32,
                                                "shape": None,
                                                "shape_c": 0,     
                                                "const_v": all_const["bias_float"],
                                                "value_file":None
                                                },      

                                "bias_add":{ "name": "bias_add",
                                            "op": "BiasAdd",
                                            "input": [fc_cfg["name_extra"] + fc_cfg["name_extra_connection"] +"QuantizedMatMulWithBiasandRelu_perChannel", "const_bias"],
                                            "attr": {
                                                        "T":{"type":"dtype", "v":dtypes.float32}
                                                }                                         
                                            },
    '''
    return _node_control_pannel



