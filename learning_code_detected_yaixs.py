import os
import csv
import pre
import pandas as pd
import MLP
import CNN
import SVM
import DWT
import kalman

# set default file path
drop_path = os.path.join('/', 'home', 'cclab-server', 'Desktop', 'detectproject', 'detected_data_learning', 'y', 'drop')  # 1
pick_path = os.path.join('/', 'home', 'cclab-server', 'Desktop', 'detectproject', 'detected_data_learning', 'y', 'pickup')  # 1
walk_path = os.path.join('/', 'home', 'cclab-server', 'Desktop', 'detectproject', 'detected_data_learning', 'y', 'walk')  # 2
run_path = os.path.join('/', 'home', 'cclab-server', 'Desktop', 'detectproject', 'detected_data_learning', 'y', 'run')  # 2
pickdown_path = os.path.join('/', 'home', 'cclab-server', 'Desktop', 'detectproject', 'detected_data_learning', 'y', 'pickdown')  # 1
situp_path = os.path.join('/', 'home', 'cclab-server', 'Desktop', 'detectproject', 'detected_data_learning', 'y', 'standup')  # 1
sitdown_path = os.path.join('/', 'home', 'cclab-server', 'Desktop', 'detectproject', 'detected_data_learning', 'y', 'sitdown')  # 1
falldown_path = os.path.join('/', 'home', 'cclab-server', 'Desktop', 'detectproject', 'detected_data_learning', 'y', 'falldown')  # 1
stop_path = os.path.join('/', 'home', 'cclab-server', 'Desktop', 'detectproject', 'detected_data_learning', 'y', 'stop')  # 1
# Load drop_files in directory
drop_files = os.listdir(drop_path)
pick_files = os.listdir(pick_path)
walk_files = os.listdir(walk_path)
run_files = os.listdir(run_path)
pickdown_files = os.listdir(pickdown_path)
situp_files = os.listdir(situp_path)
sitdown_files = os.listdir(sitdown_path)
falldown_files = os.listdir(falldown_path)
stop_files = os.listdir(stop_path)
# Load dataset in drop_files

train_input = []
train_output = []
train_output_svm = []


def learning(MLP_ch, CNN_ch, SVM_ch, axis):
    for file in drop_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        # Data Load from csv file
        x = pd.read_csv(os.path.join(drop_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(drop_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(drop_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        pre.make_shape(y)
        pre.make_shape(z)

        if axis == 'x':
            train_input.append(x.tolist())
        elif axis == 'y':
            train_input.append(y.tolist())
        elif axis == 'z':
            train_input.append(z.tolist())

        train_output.append([1, 0, 0, 0, 0, 0, 0, 0, 0])
        train_output_svm.append(0)

    for file in pick_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(pick_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(pick_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(pick_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        pre.make_shape(y)
        pre.make_shape(z)

        if axis == 'x':
            train_input.append(x.tolist())
        elif axis == 'y':
            train_input.append(y.tolist())
        elif axis == 'z':
            train_input.append(z.tolist())
        train_output.append([0, 1, 0, 0, 0, 0, 0, 0, 0])
        train_output_svm.append(1)

    for file in walk_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(walk_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(walk_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(walk_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        pre.make_shape(y)
        pre.make_shape(z)

        if axis == 'x':
            train_input.append(x.tolist())
        elif axis == 'y':
            train_input.append(y.tolist())
        elif axis == 'z':
            train_input.append(z.tolist())
        train_output.append([0, 0, 1, 0, 0, 0, 0, 0, 0])
        train_output_svm.append(2)

    for file in run_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(run_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(run_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(run_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        pre.make_shape(y)
        pre.make_shape(z)

        if axis == 'x':
            train_input.append(x.tolist())
        elif axis == 'y':
            train_input.append(y.tolist())
        elif axis == 'z':
            train_input.append(z.tolist())
        train_output.append([0, 0, 0, 1, 0, 0, 0, 0, 0])
        train_output_svm.append(3)

    for file in pickdown_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(pickdown_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(pickdown_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(pickdown_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        pre.make_shape(y)
        pre.make_shape(z)

        if axis == 'x':
            train_input.append(x.tolist())
        elif axis == 'y':
            train_input.append(y.tolist())
        elif axis == 'z':
            train_input.append(z.tolist())
        train_output.append([0, 0, 0, 0, 1, 0, 0, 0, 0])
        train_output_svm.append(4)

    for file in situp_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(situp_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(situp_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(situp_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        pre.make_shape(y)
        pre.make_shape(z)

        if axis == 'x':
            train_input.append(x.tolist())
        elif axis == 'y':
            train_input.append(y.tolist())
        elif axis == 'z':
            train_input.append(z.tolist())

        train_output.append([0, 0, 0, 0, 0, 1, 0, 0, 0])
        train_output_svm.append(5)

    for file in sitdown_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(sitdown_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(sitdown_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(sitdown_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        pre.make_shape(y)
        pre.make_shape(z)

        if axis == 'x':
            train_input.append(x.tolist())
        elif axis == 'y':
            train_input.append(y.tolist())
        elif axis == 'z':
            train_input.append(z.tolist())

        train_output.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
        train_output_svm.append(6)

    for file in falldown_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(falldown_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(falldown_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(falldown_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        pre.make_shape(y)
        pre.make_shape(z)

        if axis == 'x':
            train_input.append(x.tolist())
        elif axis == 'y':
            train_input.append(y.tolist())
        elif axis == 'z':
            train_input.append(z.tolist())

        train_output.append([0, 0, 0, 0, 0, 0, 0, 1, 0])
        train_output_svm.append(7)

    for file in stop_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(stop_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(stop_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(stop_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        pre.make_shape(y)
        pre.make_shape(z)

        if axis == 'x':
            train_input.append(x.tolist())
        elif axis == 'y':
            train_input.append(y.tolist())
        elif axis == 'z':
            train_input.append(z.tolist())

        train_output.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
        train_output_svm.append(8)

    path = ""
    if axis == 'x':
        path = 'detected/y/default model/X-axis models/'
    elif axis == 'y':
        path = 'detected/y/default model/Y-axis models/'
    elif axis == 'z':
        path = 'detected/y/default model/Z-axis models/'

    for datas in train_input:
        pre.standardization(datas)
    
    if CNN_ch == 1:
        CNN.cnn(train_input, train_output, 15, 100, path + "cnn.h5", input_shape=50)
    if MLP_ch == 1:
        MLP.mlp(train_input, train_output, 15, 100, path + "mlp.h5", input_shape=50)
    if SVM_ch == 1:
        SVM.SVM(train_input, train_output_svm, path + "svm.pkl")


def learning_DWT(MLP_ch, CNN_ch, SVM_ch, iter, axis):
    for file in drop_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        # Data Load from csv file
        x = pd.read_csv(os.path.join(drop_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(drop_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(drop_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        pre.make_shape(y)
        pre.make_shape(z)

        DWT
        reg_x = DWT.dwt_denoise(x, iter)
        reg_y = DWT.dwt_denoise(y, iter)
        reg_z = DWT.dwt_denoise(z, iter)

        if axis == 'x':
            train_input.append(reg_x.tolist())
        elif axis == 'y':
            train_input.append(reg_y.tolist())
        elif axis == 'z':
            train_input.append(reg_z.tolist())

        train_output.append([1, 0, 0, 0, 0, 0, 0, 0, 0])
        train_output_svm.append(0)

    for file in pick_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(pick_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(pick_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(pick_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        pre.make_shape(y)
        pre.make_shape(z)

        reg_x = DWT.dwt_denoise(x, iter)
        reg_y = DWT.dwt_denoise(y, iter)
        reg_z = DWT.dwt_denoise(z, iter)

        if axis == 'x':
            train_input.append(reg_x.tolist())
        elif axis == 'y':
            train_input.append(reg_y.tolist())
        elif axis == 'z':
            train_input.append(reg_z.tolist())

        train_output.append([0, 1, 0, 0, 0, 0, 0, 0, 0])
        train_output_svm.append(1)

    for file in walk_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(walk_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(walk_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(walk_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        pre.make_shape(y)
        pre.make_shape(z)

        reg_x = DWT.dwt_denoise(x, iter)
        reg_y = DWT.dwt_denoise(y, iter)
        reg_z = DWT.dwt_denoise(z, iter)

        if axis == 'x':
            train_input.append(reg_x.tolist())
        elif axis == 'y':
            train_input.append(reg_y.tolist())
        elif axis == 'z':
            train_input.append(reg_z.tolist())

        train_output.append([0, 0, 1, 0, 0, 0, 0, 0, 0])
        train_output_svm.append(2)

    for file in run_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(run_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(run_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(run_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        pre.make_shape(y)
        pre.make_shape(z)

        reg_x = DWT.dwt_denoise(x, iter)
        reg_y = DWT.dwt_denoise(y, iter)
        reg_z = DWT.dwt_denoise(z, iter)

        if axis == 'x':
            train_input.append(reg_x.tolist())
        elif axis == 'y':
            train_input.append(reg_y.tolist())
        elif axis == 'z':
            train_input.append(reg_z.tolist())

        train_output.append([0, 0, 0, 1, 0, 0, 0, 0, 0])
        train_output_svm.append(3)

    for file in pickdown_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(pickdown_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(pickdown_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(pickdown_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        pre.make_shape(y)
        pre.make_shape(z)

        reg_x = DWT.dwt_denoise(x, iter)
        reg_y = DWT.dwt_denoise(y, iter)
        reg_z = DWT.dwt_denoise(z, iter)

        if axis == 'x':
            train_input.append(reg_x.tolist())
        elif axis == 'y':
            train_input.append(reg_y.tolist())
        elif axis == 'z':
            train_input.append(reg_z.tolist())

        train_output.append([0, 0, 0, 0, 1, 0, 0, 0, 0])
        train_output_svm.append(4)

    for file in situp_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(situp_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(situp_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(situp_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        pre.make_shape(y)
        pre.make_shape(z)

        reg_x = DWT.dwt_denoise(x, iter)
        reg_y = DWT.dwt_denoise(y, iter)
        reg_z = DWT.dwt_denoise(z, iter)

        if axis == 'x':
            train_input.append(reg_x.tolist())
        elif axis == 'y':
            train_input.append(reg_y.tolist())
        elif axis == 'z':
            train_input.append(reg_z.tolist())

        train_output.append([0, 0, 0, 0, 0, 1, 0, 0, 0])
        train_output_svm.append(5)

    for file in sitdown_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(sitdown_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(sitdown_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(sitdown_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        pre.make_shape(y)
        pre.make_shape(z)

        reg_x = DWT.dwt_denoise(x, iter)
        reg_y = DWT.dwt_denoise(y, iter)
        reg_z = DWT.dwt_denoise(z, iter)

        if axis == 'x':
            train_input.append(reg_x.tolist())
        elif axis == 'y':
            train_input.append(reg_y.tolist())
        elif axis == 'z':
            train_input.append(reg_z.tolist())

        train_output.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
        train_output_svm.append(6)

    for file in falldown_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(falldown_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(falldown_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(falldown_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        pre.make_shape(y)
        pre.make_shape(z)

        reg_x = DWT.dwt_denoise(x, iter)
        reg_y = DWT.dwt_denoise(y, iter)
        reg_z = DWT.dwt_denoise(z, iter)

        if axis == 'x':
            train_input.append(reg_x.tolist())
        elif axis == 'y':
            train_input.append(reg_y.tolist())
        elif axis == 'z':
            train_input.append(reg_z.tolist())

        train_output.append([0, 0, 0, 0, 0, 0, 0, 1, 0])
        train_output_svm.append(7)

    for file in stop_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(stop_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(stop_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(stop_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        pre.make_shape(y)
        pre.make_shape(z)

        reg_x = DWT.dwt_denoise(x, iter)
        reg_y = DWT.dwt_denoise(y, iter)
        reg_z = DWT.dwt_denoise(z, iter)

        if axis == 'x':
            train_input.append(reg_x.tolist())
        elif axis == 'y':
            train_input.append(reg_y.tolist())
        elif axis == 'z':
            train_input.append(reg_z.tolist())

        train_output.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
        train_output_svm.append(8)

    for datas in train_input:
        pre.standardization(datas)
        
    path = ""
    if axis == 'x':
        path = 'detected/y/DWT model/X-axis models/'
    elif axis == 'y':
        path = 'detected/y/DWT model/Y-axis models/'
    elif axis == 'z':
        path = 'detected/y/DWT model/Z-axis models/'

    if CNN_ch == 1:
        CNN.cnn(train_input, train_output, 15, 100, path + "cnn.h5", input_shape=50)
    if MLP_ch == 1:
        MLP.mlp(train_input, train_output, 15, 100, path + "mlp.h5", input_shape=50)
    if SVM_ch == 1:
        SVM.SVM(train_input, train_output_svm, path + "svm.pkl")


def learning_kalman(MLP_ch, CNN_ch, SVM_ch, axis):
    for file in drop_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        # Data Load from csv file
        x = pd.read_csv(os.path.join(drop_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(drop_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(drop_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        kalman.kamanfilter(x)
        pre.make_shape(y)
        kalman.kamanfilter(y)
        pre.make_shape(z)
        kalman.kamanfilter(z)

        if axis == 'x':
            train_input.append(x.tolist())
        elif axis == 'y':
            train_input.append(y.tolist())
        elif axis == 'z':
            train_input.append(z.tolist())

        train_output.append([1, 0, 0, 0, 0, 0, 0, 0, 0])
        train_output_svm.append(0)

    for file in pick_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(pick_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(pick_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(pick_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        kalman.kamanfilter(x)
        pre.make_shape(y)
        kalman.kamanfilter(y)
        pre.make_shape(z)
        kalman.kamanfilter(z)

        if axis == 'x':
            train_input.append(x.tolist())
        elif axis == 'y':
            train_input.append(y.tolist())
        elif axis == 'z':
            train_input.append(z.tolist())
        train_output.append([0, 1, 0, 0, 0, 0, 0, 0, 0])
        train_output_svm.append(1)

    for file in walk_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(walk_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(walk_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(walk_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        kalman.kamanfilter(x)
        pre.make_shape(y)
        kalman.kamanfilter(y)
        pre.make_shape(z)
        kalman.kamanfilter(z)

        if axis == 'x':
            train_input.append(x.tolist())
        elif axis == 'y':
            train_input.append(y.tolist())
        elif axis == 'z':
            train_input.append(z.tolist())
        train_output.append([0, 0, 1, 0, 0, 0, 0, 0, 0])
        train_output_svm.append(2)

    for file in run_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(run_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(run_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(run_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        kalman.kamanfilter(x)
        pre.make_shape(y)
        kalman.kamanfilter(y)
        pre.make_shape(z)
        kalman.kamanfilter(z)

        if axis == 'x':
            train_input.append(x.tolist())
        elif axis == 'y':
            train_input.append(y.tolist())
        elif axis == 'z':
            train_input.append(z.tolist())
        train_output.append([0, 0, 0, 1, 0, 0, 0, 0, 0])
        train_output_svm.append(3)

    for file in pickdown_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(pickdown_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(pickdown_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(pickdown_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        kalman.kamanfilter(x)
        pre.make_shape(y)
        kalman.kamanfilter(y)
        pre.make_shape(z)
        kalman.kamanfilter(z)

        if axis == 'x':
            train_input.append(x.tolist())
        elif axis == 'y':
            train_input.append(y.tolist())
        elif axis == 'z':
            train_input.append(z.tolist())
        train_output.append([0, 0, 0, 0, 1, 0, 0, 0, 0])
        train_output_svm.append(4)

    for file in situp_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(situp_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(situp_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(situp_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        kalman.kamanfilter(x)
        pre.make_shape(y)
        kalman.kamanfilter(y)
        pre.make_shape(z)
        kalman.kamanfilter(z)

        if axis == 'x':
            train_input.append(x.tolist())
        elif axis == 'y':
            train_input.append(y.tolist())
        elif axis == 'z':
            train_input.append(z.tolist())

        train_output.append([0, 0, 0, 0, 0, 1, 0, 0, 0])
        train_output_svm.append(5)

    for file in sitdown_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(sitdown_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(sitdown_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(sitdown_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        kalman.kamanfilter(x)
        pre.make_shape(y)
        kalman.kamanfilter(y)
        pre.make_shape(z)
        kalman.kamanfilter(z)

        if axis == 'x':
            train_input.append(x.tolist())
        elif axis == 'y':
            train_input.append(y.tolist())
        elif axis == 'z':
            train_input.append(z.tolist())

        train_output.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
        train_output_svm.append(6)

    for file in falldown_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(falldown_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(falldown_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(falldown_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        kalman.kamanfilter(x)
        pre.make_shape(y)
        kalman.kamanfilter(y)
        pre.make_shape(z)
        kalman.kamanfilter(z)

        if axis == 'x':
            train_input.append(x.tolist())
        elif axis == 'y':
            train_input.append(y.tolist())
        elif axis == 'z':
            train_input.append(z.tolist())

        train_output.append([0, 0, 0, 0, 0, 0, 0, 1, 0])
        train_output_svm.append(7)

    for file in stop_files:
        if ('.csv' not in file) or ('_' in file):
            continue
        x = pd.read_csv(os.path.join(stop_path, file), header=None).iloc[:, [0]].transpose().values[0]
        y = pd.read_csv(os.path.join(stop_path, file), header=None).iloc[:, [1]].transpose().values[0]
        z = pd.read_csv(os.path.join(stop_path, file), header=None).iloc[:, [2]].transpose().values[0]

        if len(x) != 50:
            continue

        pre.make_shape(x)
        kalman.kamanfilter(x)
        pre.make_shape(y)
        kalman.kamanfilter(y)
        pre.make_shape(z)
        kalman.kamanfilter(z)

        if axis == 'x':
            train_input.append(x.tolist())
        elif axis == 'y':
            train_input.append(y.tolist())
        elif axis == 'z':
            train_input.append(z.tolist())

        train_output.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
        train_output_svm.append(8)

    for datas in train_input:
        pre.standardization(datas)
    path = ""
    if axis == 'x':
        path = 'detected/y/kalman model/X-axis models/'
    elif axis == 'y':
        path = 'detected/y/kalman model/Y-axis models/'
    elif axis == 'z':
        path = 'detected/y/kalman model/Z-axis models/'

    if CNN_ch == 1:
        CNN.cnn(train_input, train_output, 15, 100, path + "cnn.h5", input_shape=50)
    if MLP_ch == 1:
        MLP.mlp(train_input, train_output, 15, 100, path + "mlp.h5", input_shape=50)
    if SVM_ch == 1:
        SVM.SVM(train_input, train_output_svm, path + "svm.pkl")
