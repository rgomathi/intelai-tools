matmul_tb_cfg={    
                                "weight_node_name":"MatMul/b",
                                "bias_node_name":"BiasAdd/bias",
                                "input_node_name":"Placeholder",
                                "name_extra":"",
                                "name_extra_connection":"",
                                "fc_output_node_name":"Relu",
                                "to_be_removed":["MatMul","BiasAdd","Relu","MatMul/b", "BiasAdd/bias"],
                                "range_mode":  b'PRESET', #{'DYNAMIC', 'PRESET', 'MIXED'} = 'DYNAMIC'")
                                 "node_info_categary":"matmul_pc",    
                                 #"range_preset_min":-1.0,       
                                 #"range_preset_max": 6.0
                                 "range_preset_min":-0.24979891,       
                                 "range_preset_max":0.15503226 
                             }
matmul_1_5_cfg={    
                                "weight_node_name":"Const",
                                "bias_node_name":"Const_1",
                                "input_node_name":"Placeholder",
                                "name_extra":"",
                                "name_extra_connection":"",
                                "fc_output_node_name":"Relu",
                                #"to_be_removed":["Relu","BiasAdd","Const","MatMul", "Const_1"],
                                "to_be_removed":["Relu","BiasAdd","Const","MatMul", "Const_1"],
                                "range_mode":  b'PRESET', #{'DYNAMIC', 'PRESET', 'MIXED'} = 'DYNAMIC'")
                                 "node_info_categary":"split_fuse_matmul",    
                                 #"range_preset_max":6.5462027,       
                                 #"range_preset_min":-2.123853
                                #  "range_preset_max":4.51301,       
                                #  "range_preset_min":-1.64317,
                                #  "clip_method":"kl",
                                #  "clip_method":"hist_apprx" 
                                #  "clip_method":"hist_brute" 
                                #  "clip_method":"aciq" 
                                # "clip_method":"gs" 
                                "clip_method":"mmse" 
                                #  "clip_method" : "None"
                             }
fc_cfg_setting = { 
#                    "matmul_tb" :  matmul_tb_cfg,
                    "matmul_1_5" :  matmul_1_5_cfg,
#                   "wnd_hiddenlayer_0" :  wnd_latest_hiddenlayer_0_cfg,
#                   "wnd_hiddenlayer_0_2" :  proxy_tf_op_layer_MatMul_cfg_2, 
#                   "proxy_tf_op_layer_MatMul" :  proxy_tf_op_layer_MatMul_cfg, 
                      }
 
