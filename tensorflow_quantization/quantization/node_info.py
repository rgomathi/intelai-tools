from tensorflow.python.framework import dtypes
import numpy as np
from constants import generate_const

def get_nodes_control_pannel(gr, fc_cfg): 
    all_const = {
                                    "weight_node_name":fc_cfg["weight_node_name"],
                                    "bias_node_name":fc_cfg["bias_node_name"],
                                    "input_shape": None,
                                    "act_new": None,
                                    "b": None,
                                    "w_for_fp32": None,
                                    "w_for_int8": None,
                                    "w_comp": None,
                                    "w_min" : None,
                                    "w_max" : None,
                                    "a_min" : None,
                                    "a_max" : None,
                                    "b_comp": None
                                    #"bias_float": None,
                                }

    _node_infor = {
                           
                            "quantize_v2":
                            {
                                "name"  :   "quantize_v2",
                                "op"    :   "QuantizeV2",
                                "input_name_extra"  : False,
                                "input" :  [
                                            # fc_cfg["input_node_name"],
                                            "const_act_new",
                                            "const_a_min",
                                            "const_a_max",
                                           ], 
                                "attr"  :   {
                                            "T":{"type":"dtype", "v":dtypes.quint8},
                                            "mode":{"type":"string", "v":b"MIN_FIRST"},
                                            # "T":{"type":"dtype", "v":dtypes.qint8},
                                            # "mode":{"type":"string", "v":b"SCALED"},
                                            "round_mode":{"type":"string", "v":b"HALF_AWAY_FROM_ZERO"},
                                            # "round_mode":{"type":"string", "v":b"HALF_TO_EVEN"},
                                            }
                            },
                            "const_act_new":
                            {
                                "name"  : "const_act_new",
                                "op"    : "Const",
                                "type"  : dtypes.float32,
                                "shape" : None,
                                "const_v"   :   all_const["act_new"]
                            },
                            "const_a_min":
                            {
                                "name"  : "const_a_min",
                                "op"    : "Const",
                                "type"  : dtypes.float32,
                                "shape" : None,
                                "const_v"   :   all_const["a_min"]
                            },
                            "const_a_max":
                            {
                                "name"  : "const_a_max",
                                "op"    : "Const",
                                "type"  : dtypes.float32,
                                "shape" : None,
                                "const_v"   :   all_const["a_max"]
                            },
                            "quantized_mm_b_r_deq":
                            {
                                #"name"  :   "quantized_mm_b_r_deq",
                                "name"  :   fc_cfg["fc_output_node_name"],
                                "op"    :   "QuantizedMatMulWithBiasAndReluAndDequantize",
                                "input_name_extra"  : False,
                                "input" :  [
                                            "quantize_v2",
                                            "const_wt_int8",
                                            "const_b_comp",
                                            "quantize_v2:1",
                                            "quantize_v2:2",
                                            "const_wt_min",
                                            "const_wt_max",
                                            #"const_min_freezed_output",
                                            #"const_max_freezed_output"
                                           ], 
                                "attr"  :   {
                                            "T1":{"type":"dtype", "v":dtypes.quint8},
                                            # "T1":{"type":"dtype", "v":dtypes.qint8},
                                            "T2":{"type":"dtype", "v":dtypes.qint8},
                                            "Tbias":{"type":"dtype", "v":dtypes.qint32},
                                            #"Tbias":{"type":"dtype", "v":dtypes.float32},
                                            "Toutput":{"type":"dtype", "v":dtypes.float32},
                                            #"Toutput":{"type":"dtype", "v":dtypes.qint32},
                                            "input_quant_mode":{"type":"string", "v":b"MIN_FIRST"},
                                            # "input_quant_mode":{"type":"string", "v":b"SCALED"},
                                            "transpose_a":{"type":"bool", "v":False},
                                            "transpose_b":{"type":"bool", "v":False}
                                            }
                            },
                            "const_wt_int8":
                            {
                                "name"  : "const_wt_int8",
                                "op"    : "Const",
                                "type"  : dtypes.qint8,
                                "shape" : None,
                                "const_v"   :   all_const["w_for_int8"]
                            },
                            "const_b_comp":
                            {
                                "name"  : "const_b_comp",
                                "op"    : "Const",
                                "type"  : dtypes.qint32,
                                #"type"  : dtypes.float32,
                                "shape" : None,
                                "const_v"   :   all_const["b_comp"]
                            },
                            "const_wt_min":
                            {
                                "name"  : "const_wt_min",
                                "op"    : "Const",
                                "type"  : dtypes.float32,
                                "shape" : None,
                                "const_v"   :   all_const["w_min"]
                            },
                            "const_wt_max":
                            {
                                "name"  : "const_wt_max",
                                "op"    : "Const",
                                "type"  : dtypes.float32,
                                "shape" : None,
                                "const_v"   :   all_const["w_max"]
                            },
                            #"relu":
                            #{
                            #    "name"  : fc_cfg["fc_output_node_name"],
                            #    "op"    : "Relu",
                            #    "input" : ["quantized_mm_b_r_deq"],
                            #    "attr"  :   {
                            #                "T":{"type":"dtype", "v":dtypes.float32}
                            #                }
                                
                            # }
                            # "requantization_range":
                            # {
                            #     "name"  :   "requantization_range",
                            #     "op"    :   "RequantizationRange",
                            #     "input_name_extra"  : False,
                            #     "input" :  [
                            #                 "quantized_mm_b",
                            #                 "quantized_mm_b:1",
                            #                 "quantized_mm_b:2"
                            #                ], 
                            #     "attr"  :   {
                            #                 "Tinput":{"type":"dtype", "v":dtypes.qint32},
                            #                 }
                            # },
                            # "requantize":
                            # {
                            #     "name"  :   "requantize",
                            #     "op"    :   "Requantize",
                            #     "input_name_extra"  : False,
                            #     "input" :  [
                            #                 "quantized_mm_b",
                            #                 "quantized_mm_b:1",
                            #                 "quantized_mm_b:2",
                            #                 "requantization_range",
                            #                 "requantization_range:1"
                            #                ], 
                            #     "attr"  :   {
                            #                 "Tinput":{"type":"dtype", "v":dtypes.qint32},
                            #                 "out_type":{"type":"dtype", "v":dtypes.quint8},
                            #                 }
                            # },
                            # "dequantize":
                            # {
                            #     "name"  :   "dequantize",
                            #     "op"    :   "Dequantize",
                            #     "input_name_extra"  : False,
                            #     "input" :  [
                            #                 "requantize",
                            #                 "requantize:1",
                            #                 "requantize:2"

                            #                ], 
                            #     "attr"  :   {
                            #                 "mode":{"type":"string", "v":b"MIN_FIRST"},
                            #                 "T":{"type":"dtype", "v":dtypes.quint8},
                            #                 }
                            # },


                            # "addn":
                            # {
                            #     "name"  : "addn",
                            #     "op"    : "AddN",
                            #     "input" : ["matmul_fp32", 
                            #                 "dequantize"],
                            #                #"quantized_mm_b_deq"],
                            #                 #"matmul_int8"],
                            #     "attr"  :   {
                            #                 "N":{"type":"int", "v":2},
                            #                 "T":{"type":"dtype", "v":dtypes.float32}
                            #                 }
                            # },
                            # "bias_add":
                            # {
                            #     "name"  : "bias_add",
                            #     "op"    : "BiasAdd",
                            #     "input" : ["addn", "const_bias"],
                            #     "attr"  :   {
                            #                 "T":{"type":"dtype", "v":dtypes.float32}
                            #                 }
                            # },
                            # "const_bias":
                            # {
                            #     "name"  : "const_bias",
                            #     "op"    : "Const",
                            #     "type"  : dtypes.float32,
                            #     "shape" : None,
                            #     "const_v"   :   all_const["b"]
                            # },
                            # "relu":
                            # {
                            #     "name"  : fc_cfg["fc_output_node_name"],
                            #     "op"    : "Relu",
                            #     "input" : ["bias_add"],
                            #     "attr"  :   {
                            #                 "T":{"type":"dtype", "v":dtypes.float32}
                            #                 }
                                
                            # }
    
                  }

    _node_control_pannel = {
                    "Settings":{
                                "Data_Folder":"/media/data/ml/WND_SW_ROOT/tf_proxy_model_quant/models/src/"
                        },
                    "name_extra":fc_cfg["name_extra"],
                    "name_extra_connection":fc_cfg["name_extra_connection"],
                     "to_be_removed":fc_cfg["to_be_removed"],
                    "to_be_added":[_node_infor["quantize_v2"],
                                   _node_infor["const_a_min"],
                                   _node_infor["const_a_max"],
                                   _node_infor["const_act_new"],
                                   #_node_infor["quantized_mm_b_deq"],
                                   _node_infor["quantized_mm_b_r_deq"],
                                   _node_infor["const_wt_int8"],
                                   _node_infor["const_b_comp"],
                                   _node_infor["const_wt_min"],
                                   _node_infor["const_wt_max"],
                                   # _node_infor["requantization_range"],
                                   # _node_infor["requantize"],
                                   # _node_infor["dequantize"],
                                   # _node_infor["addn"],
                                   # _node_infor["bias_add"],
                                   # _node_infor["const_bias"],
                                  # _node_infor["relu"],
                                  ],
                    "to_be_updated":{
                                            }
                }
               
    generate_const(gr, all_const, _node_infor, fc_cfg["clip_method"], fc_cfg["range_preset_min"], fc_cfg["range_preset_max"])
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



