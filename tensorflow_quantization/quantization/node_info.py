from tensorflow.python.framework import dtypes
import numpy as np
from constants import generate_const

def get_nodes_control_pannel_pc(gr, fc_cfg): 
    matmul_op = "QuantizedMatMulWithBiasAndReluPerChannel"
    m_output_type=dtypes.float32
    #m_output_type=dtypes.qint32
    all_const = {
                                    "weight_node_name":fc_cfg["weight_node_name"],
                                    "bias_node_name":fc_cfg["bias_node_name"],
                                    "input_shape": None,
                                    "b": None,
                                    "w": None,
                                    "w_min": None,
                                    "w_max": None,
                                }

    _node_infor = {
                                "Quantize_activation":
                                              { 
                                                "name"              : "Quantize_activation",
                                                "op"                : "QuantizeV2", 
                                                "input_name_extra"  : False,
                                                "input"             : [ 
                                                                        fc_cfg["input_node_name"], 
                                                                        "const_a_min",
                                                                        "const_a_max"
                                                                      ],
                                                "attr"              : {
                                                                        "T":{"type":"dtype", "v":dtypes.quint8},
                                                                        "mode":{"type":"string", "v":b"MIN_FIRST"},
                                                                        #"range_preset_max":{"type":"float", "v":fc_cfg["range_preset_max"]},       
                                                                        #"range_preset_min":{"type":"float", "v":fc_cfg["range_preset_min"]}
                                                                      }
                                              },

                                "QuantizedMatMulWithBiasandRelu_perChannel":
                                              { 
                                                "name"              : "QuantizedMatMulWithBiasandRelu_perChannel",
                                                "op"                : matmul_op, 
                                                "input"             :  [
                                                                        "Quantize_activation:0",
                                                                        "const_w",
                                                                        "const_bz_qint32",
                                                                        "Quantize_activation:1",
                                                                        "Quantize_activation:2", 
                                                                        "const_w_min",
                                                                        "const_w_max"
                                                                        ],
                                                "attr"              : {
                                                                       "T1":{"type":"dtype", "v":dtypes.quint8},
                                                                       "T2":{"type":"dtype", "v":dtypes.qint8},  
                                                                       "Tbias":{"type":"dtype", "v":dtypes.qint32},  
                                                                       "Toutput":{"type":"dtype", "v":m_output_type},

                                                                    #"iVnput_quant_mode":{"type":"string", "v":b"SCALED"},                                                                
                                                                    #"quant_two_bytes":{"type":"bool", "v":True},           
                                                                    #"quant_2b_mode":{"type":"string", "v":b'Q2B_SIGNLE_SCALE'},           # {'Q2B_NO_SCALE','Q2B_SIGNLE_SCALE', 'Q2B_DOUBL_SCALE', 'Q2B_TRIPLE_SCALE'}                                                                                                                        
                                                                    "transpose_a":{"type":"bool", "v":False},              
                                                                    "transpose_b":{"type":"bool", "v":False}
                                                                       }
                                                },     

                                 "const_a_min":{ "name": "const_a_min",
                                                "op": "Const",
                                                "attr": None,
                                                "type": dtypes.float32,
                                                "shape": None,
                                                "const_v":  fc_cfg["range_preset_min"],
                                                "shape_c": 1,
                                                "value_file":None
                                                        },
                                 "const_a_max":{ "name": "const_a_max",
                                                "op": "Const",
                                                "attr": None,
                                                "type": dtypes.float32,
                                                "shape": None,
                                                "const_v":  fc_cfg["range_preset_max"],
                                                "shape_c": 1,
                                                "value_file":None
                                                        },

                                 "const_w":{ "name": "const_w",
                                                "op": "Const",
                                                "attr": None,
                                                "type": dtypes.qint8,
                                                "shape": None,
                                                "const_v":  all_const["w"],
                                                "value_file":None
                                                        },   

                                 "const_w_max":{ "name": "const_w_max",
                                                "op": "Const",
                                                "attr": None,
                                                "shape": None,
                                                "type": dtypes.float32,
                                                "const_v":  all_const["w_max"],
                                                #"shape_c": 1,         
                                                "value_file":None
                                                        },   
                                "const_w_min":{ "name": "const_w_min",
                                                "op": "Const",
                                                "shape": None,
                                                "attr": None,
                                                "type": dtypes.float32,
                                                "const_v":  all_const["w_min"],
                                                #"shape_c": 1,         
                                                "value_file":None
                                                  }   ,   
                                "const_bz_qint32":{ "name": "const_bz_qint32",
                                                "op": "Const",
                                                "attr": None,
                                                "type": dtypes.qint32,
                                                "shape": None,
                                                "const_v":  all_const["b"],
                                                "type_c": np.int32,
                                                #"type_c": np.int32,
                                                #"shape_c": all_const["input_shape"],
                                                #"flat": True,
                                                #"all_same_value": 0,
                                                #"const_v": 0.0,
                                                "value_file":None
                                                } , 
                                   "Relu":{ "name": fc_cfg["fc_output_node_name"],
                                            "extra_name": False,
                                            "input_name_extra": False,
                                            "op": "Relu",
                                            "input": [fc_cfg["name_extra"] + fc_cfg["name_extra_connection"] +"QuantizedMatMulWithBiasandRelu_perChannel"],
                                            "attr": {
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
                    "to_be_added":[_node_infor[ "Quantize_activation"],
                                        _node_infor["const_a_min"],
                                        _node_infor["const_a_max"],
                                        _node_infor["QuantizedMatMulWithBiasandRelu_perChannel"],        
                                        _node_infor["const_w"],
                                        _node_infor["const_w_min"],
                                        _node_infor["const_w_max"],
                                        _node_infor["const_bz_qint32"], 
                                        _node_infor[ "Relu"],

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
        
    return _node_control_pannel



