The code and data in this branch are the test data for evaluating accuracy of activity recognition with ensemble technique.

The train and test data of all intermittent activity data are data detected when using a z-axis based detector.

DWT.py and kalman.py: preprocessing code.

Learning data: Data used when training machine-learning models. The data in columns A, B, and C are x-axis, y-axis, and z-axis acceleration data values, respectively. Each file is one input.

Test data: Data used when testing machine-learning models. The data in columns C, D, and F are 30 x-, y-, and z-axis acceleration data values that are input into the machine-learning model input, respectively. If the file name includes g7, the data was collected using the Galaxy G7. If not, the data was collected using the Galaxy 21. There are 40 test inputs in each file.

z_axis_data_base_models: Machine-learning models trained with data obtained when running a detector algorithm based on the z-axis. These are the models used to measure performance in the USING sensor paper.

learning_code_detected _xaxis.py: machine-learing training code using data that is detected using x-axis accelerometer sensor value.

learning_code_detected _yaxis.py: machine-learing training code using data that is detected using y-axis accelerometer sensor value.

learning_code_detected _zaxis.py: machine-learing training code using data that is detected using z-axis accelerometer sensor value. --> This code is used to train machine-learning models and the models are in z_axis_data_base_models folder

Note: each learning code file has tree functions: learning / learning_kalman / learning_DWT. learning function is training code with raw data. learning_kalman function is training code with kalman filter. learning_DWT function is training code with DWT.
The first to third parameters determine whether to train MLP, 1D-CNN, or SVM. For example, if 1,0,0, only MLP is trained, if 1,1,0, MLP and 1D-CNN are trained, and if 1,1,1, MLP, 1D-CNN, and SVM are trained. The fourth parameter selects which data to use among x, y, and z axis data.



test.py: performance evaluation code that shows the confusion matrix of emsemble's result. ( If you want to this code, you must edit the path of machine-learning model folder (z_axis_base_model)and test data folder

Note: The column order in the resulting confusion matrix here is drop, pick up, walk, run, pick down, stand up, sit down, fall down, stop. This order is different from Figure 13 in the paper.

