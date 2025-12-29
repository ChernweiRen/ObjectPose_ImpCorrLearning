import os

this_folder = os.getcwd()
record_dir = this_folder + "/workspace/without_GPDM_overlap_detection_record.txt"

f = open(record_dir, "r") 
TP_TN = 0
TP = 0
T = 0
P = 0
ALL = 0

lines = f.readlines()
for line in lines:
    TP_TN += float(line.split(' ')[0])
    TP += float(line.split(' ')[1])
    T += float(line.split(' ')[2])
    P += float(line.split(' ')[3])
    ALL += float(line.split(' ')[2]) + float(line.split(' ')[3]) - 2 * float(line.split(' ')[1]) + float(line.split(' ')[0])

acc = TP_TN / ALL
precise = TP / T
recall = TP / P
F1_score = 2 * precise * recall / (precise + recall)
print("Without GPDM: ")
print("acc: ", acc, ", precise: ", precise, ", recall: ", recall, ", F1_score: ", F1_score)


this_folder = os.getcwd()
record_dir = this_folder + "/workspace/with_GPDM_overlap_detection_record.txt"

f = open(record_dir, "r") 
TP_TN = 0
TP = 0
T = 0
P = 0
ALL = 0

lines = f.readlines()
for line in lines:
    TP_TN += float(line.split(' ')[0])
    TP += float(line.split(' ')[1])
    T += float(line.split(' ')[2])
    P += float(line.split(' ')[3])
    ALL += float(line.split(' ')[2]) + float(line.split(' ')[3]) - 2 * float(line.split(' ')[1]) + float(line.split(' ')[0])

acc = TP_TN / ALL
precise = TP / T
recall = TP / P
F1_score = 2 * precise * recall / (precise + recall)
print("With GPDM: ")
print("acc: ", acc, ", precise: ", precise, ", recall: ", recall, ", F1_score: ", F1_score)