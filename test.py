import csv
import pandas as pd
import ast
import kalman
import DWT
import tensorflow as tf
import joblib
import numpy as np

##=x축 모델
x_machine_mlp_x = tf.keras.models.load_model(
    "/home/cclab-server/Desktop/detectproject/detected/z/default model/X-axis models/mlp.h5")
x_machine_cnn_x = tf.keras.models.load_model(
    "/home/cclab-server/Desktop/detectproject/detected/z/default model/X-axis models/cnn.h5")
x_machine_svm_x = joblib.load(
    "/home/cclab-server/Desktop/detectproject/detected/z/default model/X-axis models/svm.pkl")

x_machine_mlp_kalman = tf.keras.models.load_model(
    "/home/cclab-server/Desktop/detectproject/detected/z/kalman model/X-axis models/mlp.h5")
x_machine_cnn_kalman = tf.keras.models.load_model(
    "/home/cclab-server/Desktop/detectproject/detected/z/kalman model/X-axis models/cnn.h5")
x_machine_svm_kalman = joblib.load(
    "/home/cclab-server/Desktop/detectproject/detected/z/kalman model/X-axis models/svm.pkl")

x_machine_mlp_dwt = tf.keras.models.load_model(
    "/home/cclab-server/Desktop/detectproject/detected/z/DWT model/X-axis models/mlp.h5")
x_machine_cnn_dwt = tf.keras.models.load_model(
    "/home/cclab-server/Desktop/detectproject/detected/z/DWT model/X-axis models/cnn.h5")
x_machine_svm_dwt = joblib.load(
    "/home/cclab-server/Desktop/detectproject/detected/z/DWT model/X-axis models/svm.pkl")

# ====y축 모델
y_machine_mlp_x = tf.keras.models.load_model(
    "/home/cclab-server/Desktop/detectproject/detected/z/default model/Y-axis models/mlp.h5")
y_machine_cnn_x = tf.keras.models.load_model(
    "/home/cclab-server/Desktop/detectproject/detected/z/default model/Y-axis models/cnn.h5")
y_machine_svm_x = joblib.load(
    "/home/cclab-server/Desktop/detectproject/detected/z/default model/Y-axis models/svm.pkl")

y_machine_mlp_kalman = tf.keras.models.load_model(
    "/home/cclab-server/Desktop/detectproject/detected/z/kalman model/Y-axis models/mlp.h5")
y_machine_cnn_kalman = tf.keras.models.load_model(
    "/home/cclab-server/Desktop/detectproject/detected/z/kalman model/Y-axis models/cnn.h5")
y_machine_svm_kalman = joblib.load(
    "/home/cclab-server/Desktop/detectproject/detected/z/kalman model/Y-axis models/svm.pkl")

y_machine_mlp_dwt = tf.keras.models.load_model(
    "/home/cclab-server/Desktop/detectproject/detected/z/DWT model/Y-axis models/mlp.h5")
y_machine_cnn_dwt = tf.keras.models.load_model(
    "/home/cclab-server/Desktop/detectproject/detected/z/DWT model/Y-axis models/cnn.h5")
y_machine_svm_dwt = joblib.load(
    "/home/cclab-server/Desktop/detectproject/detected/z/DWT model/Y-axis models/svm.pkl")

# =====z 축 모델====
z_machine_mlp_x = tf.keras.models.load_model(
    "/home/cclab-server/Desktop/detectproject/detected/z/default model/Z-axis models/mlp.h5")

z_machine_cnn_x = tf.keras.models.load_model(
    "/home/cclab-server/Desktop/detectproject/detected/z/default model/Z-axis models/cnn.h5")
z_machine_svm_x = joblib.load(
    "/home/cclab-server/Desktop/detectproject/detected/z/default model/Z-axis models/svm.pkl")

z_machine_mlp_kalman = tf.keras.models.load_model(
    "/home/cclab-server/Desktop/detectproject/detected/z/kalman model/Z-axis models/mlp.h5")
z_machine_cnn_kalman = tf.keras.models.load_model(
    "/home/cclab-server/Desktop/detectproject/detected/z/kalman model/Z-axis models/cnn.h5")
z_machine_svm_kalman = joblib.load(
    "/home/cclab-server/Desktop/detectproject/detected/z/kalman model/Z-axis models/svm.pkl")

z_machine_mlp_dwt = tf.keras.models.load_model(
    "/home/cclab-server/Desktop/detectproject/detected/z/DWT model/Z-axis models/mlp.h5")
z_machine_cnn_dwt = tf.keras.models.load_model(
    "/home/cclab-server/Desktop/detectproject/detected/z/DWT model/Z-axis models/cnn.h5")
z_machine_svm_dwt = joblib.load(
    "/home/cclab-server/Desktop/detectproject/detected/z/DWT model/Z-axis models/svm.pkl")

file_path = "/home/cclab-server/Desktop/detectproject/apptestdata/"

data = ["drop", "fall", "pickup","putdown","standup", "sitdown","run","walk","stop"]
max_th = 1*18+0.3*9

def listtostring(list):
    str_ = "["
    for i in range(len(list)):
        if i == len(list)-1:
            str_+=str(list[i])
        else:
            str_ += str(list[i])+"/"
    return str_+"]"
            
def new_detect(detector_queue):
    detect_range=20
    detect_std=3
    index = -1
    max = 0
    for i in range(len(detector_queue) - 30):
        if (abs(detector_queue[i] - detector_queue[i+detect_range]) > detect_std):
            if max < abs(detector_queue[i] - detector_queue[i+detect_range]):
                max = abs(detector_queue[i] - detector_queue[i+detect_range])
                index = i
    return index

def svm_result(result):
    ret = []
    for data in result:
        output = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        output[data] += 0.3
        ret.append(output)
        
    return ret
def result_to_matrix(input):
    output = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0 1 4 5 6
    count = 0
    state_type = 0
    result = []
    for prob in input:
        count += 1
        output[prob.index(max(prob))] += 1
        if count == 40:
            state_type += 1
            count = 0
            result.append(output)
            return output
            output = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    return result
def list_add(list):
    output = []
    for i in range(len(list[0])):
        new_result = []
        for j in range(len(list[0][0])):
            value = 0
            for k in range(len(list)):
                value += list[k][i][j]
            new_result.append(value)
        output.append(new_result)
    return output

def stringtolist(str):
    str = str.replace('/',',')
    ls = ast.literal_eval(str)
    return ls

pathes = ["drop_test.csv", "drop_test_g7.csv",
          "pickup_test.csv", "pickup_test_g7.csv",
          "walk_test.csv", "walk_test_g7.csv",
          "run_test.csv", "run_test_g7.csv",
          "putdown_test.csv", "putdown_test_g7.csv",
          "standup_test.csv", "standup_test_g7.csv",
          "sitdown_test.csv", "sitdown_test_g7.csv",
          "fall_test.csv", "fall_test_g7.csv",
          "stop_test.csv","stop_test_g7.csv"]

result_data = []
matrixs = []
for number in range(len(pathes)):
    result_data.append(int(number/2))
    x_str = pd.read_csv(file_path+pathes[number], header=None).iloc[:, [3]].transpose().values[0]
    y_str = pd.read_csv(file_path+pathes[number], header=None).iloc[:, [4]].transpose().values[0]
    z_str = pd.read_csv(file_path+pathes[number], header=None).iloc[:, [2]].transpose().values[0]

    di = [] #Detected Index
    print(int(number/2))
    print(pathes[number])
    if int(number/2) not in [2, 3, 8]:
        for i in range(len(z_str)):
            data = stringtolist(z_str[i])
            di.append(new_detect(data))
    else:
        for i in range(len(z_str)):
            di.append(69)

    x_d = []
    y_d = []
    z_d = []
    for i in range(len(x_str)):
        x_d.append(stringtolist(x_str[i])[di[i]:di[i]+30])
        y_d.append(stringtolist(y_str[i])[di[i]:di[i]+30])
        z_d.append(stringtolist(z_str[i])[di[i]:di[i]+30])

    x_k = []
    y_k = []
    z_k = []
    for i in range(len(x_str)):
        x_data = stringtolist(x_str[i])[di[i]:di[i]+30]
        y_data = stringtolist(y_str[i])[di[i]:di[i]+30]
        z_data = stringtolist(z_str[i])[di[i]:di[i]+30]
        kalman.kamanfilter(x_data)
        kalman.kamanfilter(y_data)
        kalman.kamanfilter(z_data)
        x_k.append(x_data)
        y_k.append(y_data)
        z_k.append(z_data)


    x_dw = []
    y_dw = []
    z_dw = []
    for i in range(len(x_str)):
        x_dw.append(DWT.dwt_denoise(stringtolist(x_str[i])[di[i]:di[i]+30], 0).tolist())
        y_dw.append(DWT.dwt_denoise(stringtolist(y_str[i])[di[i]:di[i]+30], 0).tolist())
        z_dw.append(DWT.dwt_denoise(stringtolist(z_str[i])[di[i]:di[i]+30], 0).tolist())

    outputs = []

    outputs.append(x_machine_mlp_x.predict(x_d))
    outputs.append(x_machine_cnn_x.predict(x_d))
    outputs.append(svm_result(x_machine_svm_x.predict(x_d)))

    outputs.append(y_machine_mlp_x.predict(y_d))
    outputs.append(y_machine_cnn_x.predict(y_d))
    outputs.append(svm_result(y_machine_svm_x.predict(y_d)))

    outputs.append(z_machine_mlp_x.predict(z_d))
    outputs.append(z_machine_cnn_x.predict(z_d))
    outputs.append(svm_result(z_machine_svm_x.predict(z_d)))

    outputs.append(x_machine_mlp_kalman.predict(x_k))
    outputs.append(x_machine_cnn_kalman.predict(x_k))
    outputs.append(svm_result(x_machine_svm_kalman.predict(x_k)))

    outputs.append(y_machine_mlp_kalman.predict(y_k))
    outputs.append(y_machine_cnn_kalman.predict(y_k))
    outputs.append(svm_result(y_machine_svm_kalman.predict(y_k)))

    outputs.append(z_machine_mlp_kalman.predict(z_k))
    outputs.append(z_machine_cnn_kalman.predict(z_k))
    outputs.append(svm_result(z_machine_svm_kalman.predict(z_k)))

    outputs.append(x_machine_mlp_dwt.predict(x_dw))
    outputs.append(x_machine_cnn_dwt.predict(x_dw))
    outputs.append(svm_result(x_machine_svm_dwt.predict(x_dw)))

    outputs.append(y_machine_mlp_dwt.predict(y_dw))
    outputs.append(y_machine_cnn_dwt.predict(y_dw))
    outputs.append(svm_result(y_machine_svm_dwt.predict(y_dw)))

    outputs.append(z_machine_mlp_dwt.predict(z_dw))
    outputs.append(z_machine_cnn_dwt.predict(z_dw))
    outputs.append(svm_result(z_machine_svm_dwt.predict(z_dw)))

    new_result = list_add(outputs)


    matrixs.append(result_to_matrix(new_result))

    for res in new_result:
        result_data.append(listtostring(res))

res_mat_ac = []
correct = 0
for i in range(9):
    res_mat_ac.append(0)

for k in range(len(matrixs)):
    correct += matrixs[k][int(k/2)]
    for i in range(len(matrixs[k])):
        res_mat_ac[i] += matrixs[k][i]
    if (k+1) %2 == 0 and k is not 0:
        print(res_mat_ac)
        res_mat_ac = []
        for i in range(9):
            res_mat_ac.append(0)

print(correct/720)
#with open("ensemble_result_all_act.csv", 'w') as file:
#    writer = csv.writer(file)
#    writer.writerow(result_data)
