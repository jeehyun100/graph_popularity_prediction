import os
from collections import Counter

node_class_file = "./data/class_info.txt"
#result_file = "./result/epoch_85406_fcnt_596_lp_result.txt"
#result_file = "./result/50_300_128_LogisticRegressionCV_n2v_0.916_0.9771689497716896.txt"
#result_file = "./result/gcn_result_0.94.txt"
result_file = "./result/ppnp_result_0.92.txt"


with open(result_file) as f:
    nodes = {int(line.split('\t')[0].rstrip("\n")): int(line.split('\t')[1].rstrip("\n")) for line in f if
             line.rstrip() != ''}

with open(node_class_file) as f:
    nodes_ori = {int(line.split('\t')[0].rstrip("\n")): int(line.split('\t')[1].rstrip("\n")) for line in f if
             line.rstrip() != ''}

popular_centrality_node = [128, 319, 838, 328, 840, 843, 144, 720, 337, 851, 273, 917, 660, 600, 921, 476, 287, 864, 545, 164, 933, 740, 807, 871, 238, 50, 251, 767]


compare_dict_w_centrality = {item[0]: item[1] for item in nodes.items() if item[0] in popular_centrality_node}
print(compare_dict_w_centrality)
print ( Counter(compare_dict_w_centrality.values()))



print ( "all : " + str(Counter(nodes.values())))

print("original data : " + str(Counter(nodes_ori.values())))