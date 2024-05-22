import ast
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import keras
import json
import tensorflow as tf
from . import filter
from . import apps
import os
import time
from usingsensor.models import user_action as ua
from usingsensor.models import action_db

activity = ['drop', 'pickup', 'walk', 'run', 'putdown', 'standup', 'sitdown', 'fall', 'stop']

def delete(request):
    action_db.objects.all()[len(action_db.objects.all())-1].delete()
    return HttpResponse(str(len(action_db.objects.all())))

def detection(request):

    start = time.time()
    mode = request.POST['mode']

    str_x_all = request.POST['x_all']
    str_y_all = request.POST['y_all']
    str_z_all = request.POST['z_all']

    x_all = ast.literal_eval(str_x_all)
    y_all = ast.literal_eval(str_y_all)
    z_all = ast.literal_eval(str_z_all)

    x = ast.literal_eval(request.POST['x'])
    y = ast.literal_eval(request.POST['y'])
    z = ast.literal_eval(request.POST['z'])

    print("data_x = ", end="")
    print(x_all)
    print("data_y = ", end="")
    print(y_all)
    print("data_z = ", end="")
    print(z_all)

    print("z = ", end="")
    print(z)




    kalman_x = filter.kalman(x)
    kalman_y = filter.kalman(y)
    kalman_z = filter.kalman(z)

    print("kalman_z = ", end="")
    print(kalman_z)

    
    dwt_x = filter.dwt_denoise(x, 0)
    dwt_y = filter.dwt_denoise(y, 0)
    dwt_z = filter.dwt_denoise(z, 0)

    #print(z)
    #print(kalman_z)
    #print(dwt_z)

    print("==================================")
    
    results = []
    
    results.append(apps.UsingsensorConfig.x_machine_mlp_x.predict([x])[0].tolist())
    results.append(apps.UsingsensorConfig.x_machine_cnn_x.predict([x])[0].tolist())
    results.append(svm_result(apps.UsingsensorConfig.x_machine_svm_x.predict([x])[0]))
    

    results.append(apps.UsingsensorConfig.y_machine_mlp_x.predict([y])[0].tolist())
    results.append(apps.UsingsensorConfig.y_machine_cnn_x.predict([y])[0].tolist())
    results.append(svm_result(apps.UsingsensorConfig.y_machine_svm_x.predict([y])[0]))

    results.append(apps.UsingsensorConfig.z_machine_mlp_x.predict([z])[0].tolist())
    results.append(apps.UsingsensorConfig.z_machine_cnn_x.predict([z])[0].tolist())
    results.append(svm_result(apps.UsingsensorConfig.z_machine_svm_x.predict([z])[0]))
    
    ##Kalman
    results.append(apps.UsingsensorConfig.x_machine_mlp_kalman.predict([kalman_x])[0].tolist())
    results.append(apps.UsingsensorConfig.x_machine_cnn_kalman.predict([kalman_x])[0].tolist())
    results.append(svm_result(apps.UsingsensorConfig.x_machine_svm_kalman.predict([kalman_x])[0]))

    results.append(apps.UsingsensorConfig.y_machine_mlp_kalman.predict([kalman_y])[0].tolist())
    results.append(apps.UsingsensorConfig.y_machine_cnn_kalman.predict([kalman_y])[0].tolist())
    results.append(svm_result(apps.UsingsensorConfig.y_machine_svm_kalman.predict([kalman_y])[0]))

    results.append(apps.UsingsensorConfig.z_machine_mlp_kalman.predict([kalman_z])[0].tolist())
    results.append(apps.UsingsensorConfig.z_machine_cnn_kalman.predict([kalman_z])[0].tolist())
    results.append(svm_result(apps.UsingsensorConfig.z_machine_svm_kalman.predict([kalman_z])[0]))

    ##dwt
    results.append(apps.UsingsensorConfig.x_machine_mlp_dwt.predict([dwt_x])[0].tolist())
    results.append(apps.UsingsensorConfig.x_machine_cnn_dwt.predict([dwt_x])[0].tolist())
    results.append(svm_result(apps.UsingsensorConfig.x_machine_svm_dwt.predict([dwt_x])[0]))

    results.append(apps.UsingsensorConfig.y_machine_mlp_dwt.predict([dwt_y])[0].tolist())
    results.append(apps.UsingsensorConfig.y_machine_cnn_dwt.predict([dwt_y])[0].tolist())
    results.append(svm_result(apps.UsingsensorConfig.y_machine_svm_dwt.predict([dwt_y])[0]))

    results.append(apps.UsingsensorConfig.z_machine_mlp_dwt.predict([dwt_z])[0].tolist())
    results.append(apps.UsingsensorConfig.z_machine_cnn_dwt.predict([dwt_z])[0].tolist())
    results.append(svm_result(apps.UsingsensorConfig.z_machine_svm_dwt.predict([dwt_z])[0]))
    
    each = [0,0,0,0,0,0,0,0,0]
    for r in results:
        each[r.index(max(r))] += 1
    #print(each)

    #ensemble

    ensemble_result = list_add(results)

    act = activity[ensemble_result.index(max(ensemble_result))]

    print(mode == "monitor")

    if mode == "monitor" and act not in ['walk', 'run', 'stop']:
        print("Act error")
        return HttpResponse("act error")

    if mode == "detector" and act in ['walk', 'run', 'stop']:
        print("Act error")
        return HttpResponse("act error")


    print("classification result: " + act)

    #new_x_all = str_x_all.replace(',', '/')
    #new_y_all = str_y_all.replace(',', '/')
    #new_z_all = str_z_all.replace(',', '/')
    #act_db = action_db(action=act, x_data=new_x_all, y_data=new_y_all, z_data=new_z_all)
    #act_db.save()

    #numofdata = str(len(action_db.objects.all()))

    #return HttpResponse(act + "/"+numofdata)


    if not(ua.objects.all()):
        db = ua(action=act, decouple="couple")
        db.save()
    else:
        db = ua.objects.all()[0]
        #bef_act = db.action
        #if state_filter(bef_act, act, mode) == "x":
        #    print("activity error" + act)
        #    return HttpResponse("act error")
        #print(act)

        state = db.decouple

        new_state = check_decouple(act, state)
        print(new_state)
        db.action = act
        db.decouple = new_state
        db.save()

    return HttpResponse(new_state)




    

def list_add(list):
    output = []
    for i in range(len(list[0])):
        value = 0
        for k in range(len(list)):
            value += list[k][i]
        output.append(value)
    print(output)
    return output
    
def svm_result(result):
    output = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    output[result] += 0.3

    return output

#activity = ['drop', 'pickup', 'walk', 'run', 'putdown', 'standup', 'sitdown', 'fall', 'stop']
def check_decouple(action, state):
    c1 = ["walk", "run", "standup", "sitdown", "stop"]
    c2 = ["fall", "putdown", "drop"]
    c3 = ["pickup", "standup"]
    c4 = ["stop"]

    if state == "couple":
        if action in c1:
            return "couple"
        elif action in c2:
            return "decouple"
        else:
            return state
    if state == "decouple":
        if action in c3:
            return "couple"
        elif action in c4:
            return "decouple"
        else:
            return state




def state_filter(bef, aft, mode):
    if bef == "drop":
        if aft not in ["stop", "pickup"]:
            return "x"
    if bef == "putdown":
        if aft not in ["stop", "pickup"]:
            return "x"
    if bef == "pickup":
        if aft == "pickup":
            return "x"
    if bef == "sitdown":
        if aft in ["pickup", "sitdown","fall", "walk", "run"]:
            return "x"
    if bef == "standup":
        if aft in ["pickup", "standup"]:
            return "x"
    if bef == "falldown":
        if aft not in ["stop", "pickup", "standup"]:
            return "x"
    if bef == "walk" or bef == "run":
        if aft not in ["drop", "fall", "walk", "run", "stop"]:
            return "x"
    return "o"




# Create your views here.


