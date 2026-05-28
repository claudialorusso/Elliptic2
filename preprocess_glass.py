import pandas as pd
import torch
import random
import time
import pickle

preprocessing_start = time.time()
#Parameters
DATAPATH = "./dataset/"
train = 0.8 #percentage of training subgraph
val = 0.1 #percentage of validation subgraph

#Read in nodes
start = time.time()
feat = pd.read_csv(DATAPATH+"/background_nodes.csv")
print(list(feat.columns[1:]))
print(feat.head())
print("load background node time", time.time()-start)
print("total number of node: ",len(feat))
start = time.time()

#Read in Node ID
n2id = {}
maxid = 0
for row in feat.itertuples(index=True):
    n2id[row[1]] =int(row[0])
    maxid = max(int(row[0]),maxid)

print("store all nodes", time.time()-start)
start = time.time()
with open('n2id.pkl', 'wb') as fp:
    pickle.dump(n2id, fp)

#Read in edge list
edge = pd.read_csv(DATAPATH+"/background_edges.csv",usecols=["clId1","clId2"])
print("load background edge time", time.time()-start)
print("total number of edge: ",len(edge))
start = time.time()

file = open("./edge_list.txt","w")
for t in edge.itertuples(index=False):
    (c1,c2) = t
    if n2id[c1] > maxid:
        print("WARNING NODE OUT OF RANGE:",c1,n2id[c1])
    if n2id[c2] > maxid:
        print("WARNING NODE OUT OF RANGE:",c2,n2id[c1])
    file.write(str(n2id[c1])+" "+str(n2id[c2])+"\n")
file.close()
print("time to store edgelist", time.time()-start)


#Read in Subgraph

start = time.time()
cc = pd.read_csv(DATAPATH+"/connected_components.csv")
edge = pd.read_csv(DATAPATH+"/edges.csv")
node = pd.read_csv(DATAPATH+"/nodes.csv")
print("load rest time", time.time()-start)
start = time.time()

#Read in Subgraph ID
cc2id = {}
c=0
for row in cc.itertuples(index=True):
    cc2id[row[1]] =int(row[0])
    c+=1
print("number of subgraph ",c)

sub = {}
for row in node.itertuples(index=False):
    if cc2id[row[1]] in sub.keys():
        sub[cc2id[row[1]]] += "-"+str(n2id[row[0]])
    else:
        sub[cc2id[row[1]]] = str(n2id[row[0]])

#Generate Subgraph.pth
file = open("./subgraphs.pth","w")

#fair and customised splitting policy
# NB since it does caching, if swapping between the two kinds of splitting methods, delete :train_sub_G.pt val_sub_G.pt and test_sub_G.pt
train_count = 0
val_count = 0
test_count = 0
for i in sub.keys():
    label = cc.loc[i, "ccLabel"]
    split_value = cc.loc[i, "split"]
    if split_value == "TRN":
        split_str = "train"
        train_count += 1
    elif split_value == "VAL":
        split_str = "val"
        val_count += 1
    elif split_value == "TST":
        split_str = "test"
        test_count += 1
    else:
        continue
    file.write(sub[i] + "\t" + label + "\t" + split_str + "\n")

print("TRAIN:", train_count)
print("VAL:", val_count)
print("TEST:", test_count)
"""
# original splitting policy
counter = 0
for i in sub.keys():
    counter += 1
    label = cc.loc[i,"ccLabel"]
    if counter%10 <=7:
        file.write(sub[i]+"\t"+label+"\t"+"train\n")
    elif counter%10 ==8:
        file.write(sub[i]+"\t"+label+"\t"+"val\n")
    else:
        file.write(sub[i]+"\t"+label+"\t"+"test\n")
"""
file.close()
print("generate subgraph.pth time: ", time.time()-start)
tot_time = time.time() - preprocessing_start
print(f"\033[33mTotal runtime: {tot_time:.2f}s\033[0m")
print(f"\033[33mTotal runtime: {tot_time/ 60:.2f}min\033[0m")
