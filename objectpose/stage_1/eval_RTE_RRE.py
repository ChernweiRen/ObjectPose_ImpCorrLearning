import numpy as np
import os

this_folder = os.getcwd()
record_dir = this_folder + "/workspace/RTE_RRE_record.txt"

f = open(record_dir, "r") 

RTE = []
RRE = []
lines = f.readlines()
for line in lines:
    T = line.split('m ')[0]
    R = line.split('m')[1]
    RTE.append(float(T))
    R = R.replace('°','')
    RRE.append(float(R))


RTE = np.array(RTE)
RRE = np.array(RRE)
mean_RTE = np.mean(RTE)
mean_RRE = np.mean(RRE)
std_RTE = np.std(RTE)
std_RRE = np.std(RRE)


RTE_mask = (RTE < 2)
RRE_mask = (RRE < 5)
acc_mask = (RTE_mask & RRE_mask)
acc = (acc_mask.sum(-1) / acc_mask.shape[0])
print("mean_RRE: ", mean_RRE, ", mean_RTE: ", mean_RTE, ", std_RRE: ", std_RRE, ", std_RTE: ", std_RTE, ", Acc.", acc*100)
