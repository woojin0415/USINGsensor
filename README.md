The train and test data of all intermittent activity data are data detected when using a z-axis based detector.

Learning data: Data used when training machine-learning models. The data in columns A, B, and C are x-axis, y-axis, and z-axis acceleration data values, respectively. Each file is one input.

Test data: Data used when testing machine-learning models. The data in columns C, D, and F are 30 x-, y-, and z-axis acceleration data values that are input into the machine-learning model input, respectively. If the file name includes g7, the data was collected using the Galaxy G7. If not, the data was collected using the Galaxy 21. There are 40 test inputs in each file.

z_axis_data_base_models: Machine-learning models trained with data obtained when running a detector algorithm based on the z-axis. These are the models used to measure performance in the USING sensor paper.
