import collections
import re
import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import app
from tensorflow.python.platform import flags as flags_lib
from tensorflow.python.platform import gfile
from google.protobuf import text_format

from node_info  import  get_nodes_control_pannel
from fc_cfg import fc_cfg_setting

__func_categary = {
         "split_fuse_matmul": get_nodes_control_pannel,
#         "2b_qint8_matmul": get_nodes_control_pannel_2b_qint8,
#         "weight_expand": get_nodes_control_pannel_weight_expand,
#         "we_qint8_matmul": get_node_info_we_qint8_matmul,
#         "qint8_2b_matmul": get_node_info_we_qint8_h2b_matmul,
#         "qint8_2b_matmul_org": get_node_info_we_qint8_h2b_matmul_org,
   }

def get_nodes_control_pannel(gr, model_cfg_name, mode): 
    if model_cfg_name in fc_cfg_setting.keys():
        fc_cfg=fc_cfg_setting[model_cfg_name]
    else:
        print("Error,  the model configuration is not supported.")
        print("Please check model_cfg_name", model_cfg_name)
        return None

        
    if "node_info_categary" in fc_cfg.keys():
        print(fc_cfg)
        print("===========>", model_cfg_name, "<===========")
        print("===========>", fc_cfg, "<===========")
        if fc_cfg["node_info_categary"] in __func_categary.keys() and __func_categary[fc_cfg["node_info_categary"]] is not None:
            return __func_categary[fc_cfg["node_info_categary"]](gr, fc_cfg)
        print("Error,  node_info_func is not defined.")
        print("Please check ", model_cfg_name, "settings.")
#    elif mode == "two_bytes_quant":
#        print("===========>", model_cfg_name, "<===========")
#        return get_nodes_control_pannel_2b_quint8(gr, fc_cfg)
        
    return None
        
        
    
