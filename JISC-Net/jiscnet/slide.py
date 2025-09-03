import numpy
import math
import torch

def slide_MTS_dim(X, alpha):
    num_of_old_MTS = numpy.shape(X)[0]
    dim_of_old_MTS = numpy.shape(X)[1]
    length_of_old_MTS = numpy.shape(X)[2]
    length_of_new_MTS = int(length_of_old_MTS * alpha)

    X_alpha = X[:, :, 0 : length_of_new_MTS]

    # determine step
    if (length_of_old_MTS <= 50) :
        step = 1
    elif (length_of_old_MTS > 50 and length_of_old_MTS <= 100):
        step = 2
    elif (length_of_old_MTS > 100 and length_of_old_MTS <= 300):
        step = 3
    elif (length_of_old_MTS > 300 and length_of_old_MTS <= 1000):
        step = 4
    elif (length_of_old_MTS > 1000 and length_of_old_MTS <= 1500):
        step = 5
    elif (length_of_old_MTS > 1500 and length_of_old_MTS <= 2000):
        step = 7
    elif (length_of_old_MTS > 2000 and length_of_old_MTS <= 3000):
        step = 10
    else:
        step = 100

    # determine step number
    step_num = int(math.ceil((length_of_old_MTS -length_of_new_MTS)/step))

    # still slide to 3D
    for k in range(1, length_of_old_MTS -length_of_new_MTS+1, step):
        X_temp = X[:, :, k : length_of_new_MTS + k]
        X_alpha = numpy.concatenate((X_alpha, X_temp), axis = 0)

    X_alpha = X_alpha.reshape(num_of_old_MTS * (step_num+1), dim_of_old_MTS, length_of_new_MTS) #(7833, 3, 60)

    return X_alpha



def slide_MTS_dim_step(X, class_label, alpha): # X_alpha:(sliding 갯수, 변수 수, 시계열 길이), candidate_class_label:(sliding 갯수)
    #X: (373, 3, 100)
    #class_label: (373,)
    num_of_old_MTS = numpy.shape(X)[0] #373
    dim_of_old_MTS = numpy.shape(X)[1] # 3
    length_of_old_MTS = numpy.shape(X)[2] #100
    length_of_new_MTS = int(length_of_old_MTS * alpha) #60

    X_alpha = X[:, :, 0 : length_of_new_MTS] #(373, 3, 60)

    # determine step
    if (length_of_old_MTS <= 50) :
        step = 1
    elif (length_of_old_MTS > 50 and length_of_old_MTS <= 100):
        step = 2
    elif (length_of_old_MTS > 100 and length_of_old_MTS <= 300):
        step = 3
    elif (length_of_old_MTS > 300 and length_of_old_MTS <= 1000):
        step = 4
    elif (length_of_old_MTS > 1000 and length_of_old_MTS <= 1500):
        step = 5
    elif (length_of_old_MTS > 1500 and length_of_old_MTS <= 2000):
        step = 6
    elif (length_of_old_MTS > 2000 and length_of_old_MTS <= 3000):
        step = 7
    else:
        step = 100

    # determine step number
    step_num = int(math.ceil((length_of_old_MTS - length_of_new_MTS)/step)) #20 -> ...

    # still slide to 3D
    for k in range(1, length_of_old_MTS -length_of_new_MTS+1, step):
        X_temp = X[:, :, k : length_of_new_MTS + k]
        X_alpha = numpy.concatenate((X_alpha, X_temp), axis = 0)

    X_alpha = X_alpha.reshape(num_of_old_MTS * (step_num+1), dim_of_old_MTS, length_of_new_MTS) #(7833, 3, 60)
    # compose the final class label
    candidate_class_label = numpy.tile(class_label, step_num+1) #(7833,) 반복

    return X_alpha, candidate_class_label




def slide_MTS_tensor_step(X, alpha):
    num_of_old_MTS = X.size(0) #8
    dim_of_old_MTS = X.size(1) #3
    length_of_old_MTS = X.size(2) #100
    length_of_new_MTS = int(length_of_old_MTS * alpha) #60

    X_alpha = X[:, :, 0 : length_of_new_MTS].unsqueeze(1) #torch.Size([8, 1, 3, 60])

    # determine step
    if (length_of_old_MTS <= 50) :
        step = 1
    elif (length_of_old_MTS > 50 and length_of_old_MTS <= 100):
        step = 2
    elif (length_of_old_MTS > 100 and length_of_old_MTS <= 300):
        step = 3
    elif (length_of_old_MTS > 300 and length_of_old_MTS <= 1000):
        step = 4
    elif (length_of_old_MTS > 1000 and length_of_old_MTS <= 1500):
        step = 5
    elif (length_of_old_MTS > 1500 and length_of_old_MTS <= 2000):
        step = 6
    elif (length_of_old_MTS > 2000 and length_of_old_MTS <= 3000):
        step = 7
    else:
        step = 100

    # determine step number
    step_num = int(math.ceil((length_of_old_MTS - length_of_new_MTS)/step)) #반올림 #20

    # still slide to 3D
    for k in range(1, length_of_old_MTS-length_of_new_MTS+1, step):
        X_temp = X[:, :, k : length_of_new_MTS + k] #torch.Size([8, 3, 60])
        X_temp = X_temp.unsqueeze(1)  # X_temp의 shape을 (1, 8, 3, 100)으로 변경
        X_alpha = torch.cat((X_alpha, X_temp), dim=1)  # X_alpha와 X_temp를 dim=0 차원에서 결합

    X_beta = torch.reshape(X_alpha, (num_of_old_MTS, (step_num+1), dim_of_old_MTS, length_of_new_MTS)) #fix
    
    return X_beta
