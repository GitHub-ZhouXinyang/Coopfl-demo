#!/usr/bin/env python
import socket
import time
import torch
import sys
import time
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np
import argparse
import torchvision
import torch.nn.functional as F
import random
import os
import socket
import threading
import time
import struct
from util.utils import send_msg, recv_msg, time_printer,add_model, scale_model, printer_model, time_duration
import copy
from torch.autograd import Variable
from model.model_VGG_cifar import construct_VGG_cifar
from model.model_AlexNet_cifar import construct_AlexNet_cifar
from model.model_nin_cifar import construct_nin_cifar
from model.model_VGG_image import construct_VGG_image
from model.model_AlexNet_image import construct_AlexNet_image
from model.model_nin_image import construct_nin_image
from model.model_nin_emnist import construct_nin_emnist
from model.model_AlexNet_emnist import construct_AlexNet_emnist
from model.model_VGG_emnist import construct_VGG_emnist
from util.utils import printer
import math
import numpy.ma as ma
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

parser = argparse.ArgumentParser(description='PyTorch MNIST SVM')
parser.add_argument('--device_num', type=int, default=1, metavar='N',
                        help='number of working devices ')
parser.add_argument('--edge_number', type=int, default=1, metavar='N',
                        help='edge server')
parser.add_argument('--model_type', type=str, default='NIN', metavar='N',          #NIN,AlexNet,VGG
                        help='model type')
parser.add_argument('--dataset_type', type=str, default='cifar10', metavar='N',  #cifar10,cifar100,image,emnist
                        help='dataset type')
args = parser.parse_args()

if True:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)
device_gpu = torch.device("cuda" if True else "cpu")
   
lr = 0.01
device_num = args.device_num
edge_num = args.edge_number
model_length = 0
delay_gap = 10
epoch_max = 500
acc_count = []
criterion = nn.NLLLoss()

listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind(('localhost', 50010))
#listening_sock.bind(('172.16.50.22', 50010))

listening_sock1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock1.bind(('localhost', 50002))
#listening_sock1.bind(('172.16.50.22', 50011))

edge_sock_all = []

while len(edge_sock_all) < edge_num:
    listening_sock1.listen(edge_num)
    print("Waiting for incoming connections...")
    (client_sock, (ip, port)) = listening_sock1.accept()
    print('Got connection from ', (ip,port))
    print(client_sock)
    edge_sock_all.append(client_sock)

device_sock_all = [None]*device_num
#connect to device
for i in range(device_num):
    listening_sock.listen(device_num)
    print("Waiting for incoming connections...")
    (client_sock, (ip, port)) = listening_sock.accept()
    msg = recv_msg(client_sock)
    print('Got connection from node '+ str(msg[1]))
    print(client_sock)
    device_sock_all[msg[1]] = client_sock





#test the accuracy of model after aggregation
def test(models, dataloader, dataset_name, epoch, start_time):
    for model in models:
        model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            x=data.to(device_gpu)
            for i in range(0,len(models)):
                y = models[i](x)
                if i<len(models)-1:
                    x = y
                else:
                    loss += criterion(y, target)
                    pred = y.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
    end_time = time.time()
    a,b = time_duration(start_time, end_time)
    printer("Epoch {} Duration {}s {}ms Testing loss: {}".format(epoch,a,b,loss/len(dataloader)))
    printer("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                correct,
                                                len(dataloader.dataset),
                                                100. * correct / len(dataloader.dataset)))
    acc_count.append(correct*1.0/len(dataloader.dataset))

if args.dataset_type == 'cifar100':
    print("cifar100")
    transform = transforms.Compose([ 
                               #     transforms.RandomCrop(32, padding=4),
                                #    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/test_cifar100', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

elif args.dataset_type == 'cifar10':
    transform = transforms.Compose([ 
                                #    transforms.RandomCrop(32, padding=4),
                                #    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/cifar10/test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

elif args.dataset_type == 'emnist':
    transform = transforms.Compose([
                           transforms.Resize(28),
                           #transforms.CenterCrop(227),
                           transforms.Grayscale(1),
                           transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
          ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/emnist/byclass_test', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)

elif args.dataset_type == 'image' and args.model_type != "AlexNet":
    transform = transforms.Compose([
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/IMAGE10/test', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)

elif args.dataset_type == 'image' and args.model_type == "AlexNet":
    transform = transforms.Compose([  transforms.Scale((227,227)),
                               #   transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                              ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/IMAGE10/test', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)

#get the information about edge and device
def Get_commp_comm_mem_of_device_edge(str,epoch):
    compute_device = []
    communication_device = []
    memory_device = []
    data_size = []
    for i in range(device_num):
        msg = recv_msg(device_sock_all[i],"CLIENT_TO_SERVER") #compute, communication, memory,data_size 0-device1,1-device2...
        compute_device.append(msg[1])
        communication_device.append(msg[2])
        memory_device.append(msg[3])
        data_size.append(msg[4])
    compute_edge = []
    communication_edge = []
    memory_edge = []
    for i in range(edge_num):
        msg = recv_msg(edge_sock_all[i],"CLIENT_TO_SERVER") #compute, communication, memory
        compute_edge.append(msg[1])
        communication_edge.append(msg[2])
        memory_edge.append(msg[3])
    if str == 'NIN' and args.dataset_type == "image":
        model_transmission = [70.90,70.90,70.90,17.09,45.56,45.56,45.56,10.56,15.84,15.84,15.84,3.38,9.00,9.00,0.09,0]
        model_size = [1.0708,0.288,0.28,0,18.76,2.0170,2.0170,0,27.02,4.524,4.524,0,108.06,32.0639,0.3129,0]
        layer_memory = 2*(model_size + model_transmission)
    elif str == 'VGG' and args.dataset_type == "image":
        model_transmission = [784,784,196,392,392,98,196,196,196,49,98,98,98,24.5,24.5,24.5,24.5,6.13,1,1,0]
        model_size = [0.008,0.14,0,0.28,0.56,0,1.13,2.25,2.25,0,4.5,9.01,9.01,0,9.01,9.01,9.01,0,392.02,64.02,15.63]
        layer_memory = 2*(model_size + model_transmission)
    elif str == 'AlexNet' and args.dataset_type == "image":
        model_transmission = [70.9,17.09,45.56,10.56,15.84,15.84,10.56,2.25,1,1,0.24]
        model_size = [0.133,0,2.345,0,3.377,5.054,3.377,0,144.016,64.016,15.629,0]
        layer_memory = 2*(model_size + model_transmission)
    elif str == 'NIN' and args.dataset_type != "image":
        if args.dataset_type == 'emnist':
            model_transmission = [36.75,30.6,18.38,4.59,9.19,9.19,9.19,2.3,2.3,2.3,0.74,0]
            model_size = [0.056,0.11,0.05,0,1.759,0.142,0.1422,0,1.267,0.1422,0.007,0]
        else:
            model_transmission = [48.0,40.0,24.0,6.0,12.0,12.0,12.0,3.0,3.0,3.0,0.16,0]
            model_size = [0.056,0.11,0.05,0,1.759,0.142,0.1422,0,1.267,0.1422,0.007,0]
        layer_memory = 2*(model_size + model_transmission)
    elif str == 'VGG' and args.dataset_type != "image":
        model_transmission = [16.0,16.0,4.0,8.0,8.0,2.0,4.0,4.0,4.0,1.0,2.0,2.0,2.0,0.5,0.5,0.5,0.5,0.13,1.00,1.0,0.0]
        model_size = [0.133,0,2.345,0,3.377,5.054,3.377,0,144.016,64.016,15.629,0,0,0,0,0,0,0,0,0]
        layer_memory = 2*(model_size + model_transmission)
    elif str == 'AlexNet' and args.dataset_type != "image":
        model_transmission = [4.0,1.0,3.0,0.75,1.5,1,1,0.25,1,1,0.002]
        model_size = [0.133,0,2.345,0,3.377,5.054,3.377,0,144.016,64.016,15.629,0]
        layer_memory = 2*(model_size + model_transmission)
    partiiton_way, offloading_descision = partition_algorithm()
    return partiiton_way, offloading_descision



def partition_algorithm():
   # if epoch == 0 or epoch >4:
    if True:
        partition_way = []
        offloading_descision = []
        for i in range(device_num):
            partition_way.append([])
            partition_way[i].append(0)
            for j in range(model_length-1):
                partition_way[i].append(0)
         #   pooling_layer = [2,5,9,13,17]
         #   pooling_layer = [1,3,7]
            # random.shuffle(pooling_layer)
            # a = pooling_layer[0]
            # for j in range(1, a+1):
            #     partition_way[i].append(0)
            # for j in range(a+1,model_length):
            #     partition_way[i].append(1)
            
              #  partition_way[i].append(random.randint(0,1))
        for i in range(device_num):
            offloading_descision.append(i % edge_num+1)
    return partition_way, offloading_descision


def send_msg_to_device_edge(sock_adr, msg):
    send_msg(sock_adr, msg)


#cionstruct the part of model for each node
def part_model_construct(partition_way, models):
    node_models = [None]*len(models)
    for i in range(len(models)):
        if partition_way[i] == 0:
            node_models[i] = copy.deepcopy(models[i])
    return node_models


#initation the model and send to devices and edges
def model_send_with_partition(partiiton_way,offloading_descision,models):
    send_device=[]
    for i in range(device_num):
      #  model_device = part_model_construct(partiiton_way[i], models)
        msg = ['SERVER_TO_CLIENT', partiiton_way[i], offloading_descision]
        send_device_msg = threading.Thread(target=send_msg_to_device_edge, args=(device_sock_all[i], msg))
        send_device_msg.start()

       # send_msg(device_sock_all[i],msg)

    for i in range(edge_num):
        msg = ['SERVER_TO_CLIENT', models ,partition_way, offloading_descision, time.time()]
        #send_msg(edge_sock_all[i], msg )
        send_edge_msg = threading.Thread(target=send_msg_to_device_edge, args=(edge_sock_all[i], msg))
        send_edge_msg.start()


#the algorithm stops when accuauracy of changed less than 2% in 10 epochs 
def train_stop():
    if len(acc_count)<11:
        return False
    max_acc = max(acc_count[len(acc_count)-10:len(acc_count)])
    min_acc = min(acc_count[len(acc_count)-10:len(acc_count)])
    if max_acc-min_acc <=0.001:
        return True
    else:
        return False


def rev_msg_edge(sock,epoch,edge_id,offloading_descision):
    global rec_models
    global rec_time
    msg = recv_msg(sock,"CLIENT_TO_SERVER")
  #  models = copy.deepcopy(scale_model(msg[1],offloading_descision.count(edge_id)/len(offloading_descision)))
    if offloading_descision.count(edge_id)!=0:
        rec_models.append(scale_model(msg[1],float(offloading_descision.count(edge_id)/len(offloading_descision))))
        rec_time[epoch].append(time.time()-msg[2]+msg[3])
    print(msg[3], offloading_descision.count(edge_id)/len(offloading_descision))
 #   rec_time[epoch].append(time.time()-msg[2]+msg[3])

rec_models = []
rec_time = []
communication_cost=0
#models, optimizers = construct_VGG([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
#models, optimizers = construct_AlexNet([0,0,0,0,0,0,0,0,0,0,0],lr)
if args.dataset_type == "image":
    if args.model_type == "NIN":
        models, optimizers = construct_nin_image([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 16
    elif args.model_type == "AlexNet":
        models, optimizers = construct_AlexNet_image([0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 11
    elif args.model_type == "VGG":
        models, optimizers = construct_VGG_image([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 21
elif args.dataset_type == 'emnist':
    if args.model_type == "NIN":
        models, optimizers = construct_nin_emnist([0,0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 12
    elif args.model_type == "AlexNet":
        models, optimizers = construct_AlexNet_emnist([0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 11
    elif args.model_type == "VGG":
        models, optimizers = construct_VGG_emnist([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 21
else:
    if args.model_type == "NIN":
        models, optimizers = construct_nin_cifar([0,0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 12
    elif args.model_type == "AlexNet":
        models, optimizers = construct_AlexNet_cifar([0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 11
    elif args.model_type == "VGG":
        models, optimizers = construct_VGG_cifar([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 21


for model in models:
    model_size = 0
    count = 0
    if model!=None:
        for para in model.parameters():
            model_size+=sys.getsizeof(para.storage())/(1024*1024)
        print("layer " +str(count) + "model size " +str(model_size)+"MB")
        count+=1
start_time = time.time()
for epoch in range(epoch_max):
    rec_time.append([])
    print(acc_count)
    partition_way, offloading_descision = Get_commp_comm_mem_of_device_edge(args.model_type,epoch)
    printer("partition_way_and_offloading_descision {},{} ".format(partition_way,offloading_descision))
    model_send_with_partition(partition_way, offloading_descision, models)
  #  print("epoch"+str(epoch)+'before update'+'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
  #  for model in models:
   #     for para in model.parameters():
        #    printer_model(para)
    rev_msg_d = []
    for i in range(edge_num):
       # msg = recv_msg(device_sock_all[i],"CLIENT_TO_SERVER") #get the parameter [0,weight]
        print("rec models")
        rev_msg_d.append(threading.Thread(target = rev_msg_edge, args = (edge_sock_all[i],epoch, i+1,offloading_descision)))
        rev_msg_d[i].start()
    for i in range(edge_num):
        rev_msg_d[i].join()
    for i in range(1,len(rec_models)):
        rec_models[0] = copy.deepcopy(add_model(rec_models[0], rec_models[i]))
    models = copy.deepcopy(rec_models[0])
    rec_models.clear()
    test(models, testloader, "Test set", epoch, start_time)
  #  print(rec_time)
    for i in range(len(rec_time[epoch])):
        communication_cost += rec_time[epoch][i]/len(rec_time[epoch])
    printer("Model_distribution_and_collection_time {} ".format(communication_cost))
  #  print("epoch"+str(epoch)+'before update'+'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#    for model in models:
      #  for para in model.parameters():
         #   printer_model(para)
    if train_stop():
        break

print("The traing process is over")


 
#tensor([ 0.0034, -0.0065, -0.0107, -0.0122, -0.0101, -0.0042, -0.0142, -0.0101,
    #     0.0074, -0.0019], requires_grad=True)
#tensor([ 0.0031, -0.0064, -0.0105, -0.0122, -0.0097, -0.0043, -0.0140, -0.0100,
 #        0.0069, -0.0021], requires_grad=True)

