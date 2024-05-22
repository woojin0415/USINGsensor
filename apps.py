from django.apps import AppConfig
import tensorflow as tf
import joblib

class UsingsensorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'usingsensor'

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)



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

    
    

    #====y축 모델
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
    

    
    
    #=====z 축 모델====
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



        
