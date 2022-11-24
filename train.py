import argparse
import pandas as pd
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import random_split
import torch.utils.data.distributed
from dataset_new import TTDataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.optim as optim
import torch
import torch.nn as nn
from model import EventSpotting
from utils import resize_around_coords, coordinates_constructor
from PIL import Image
from tqdm import tqdm




from sklearn import metrics
import seaborn as snNew
from matplotlib import pyplot as plt
import pickle as pkl
     

def confusion_matrix_plot(label, gt):

     cm_train = metrics.confusion_matrix(gt,label)
     DetaFrame_cm = pd.DataFrame(cm_train, range(4), range(4))
     snNew.heatmap(DetaFrame_cm, annot=True)
     snNew.set(rc = {'figure.figsize':(20,20)})
     plt.show()


## not needed here, or i can call the training function from the other file. ba'd mein daikhty hain
def train_gs_block(device, model, train_loader, learning_rate, criterion, optimizer,epochs,len):

     gt = list()
     ed_output = list()
     running_loss = 0
     correct = 0
     loss_epoch = list()
     accruracy_epoch = list()

     for epoch in range(1,epochs+1):
          e_loss = 0
          e_acc = 0
          correct_samples = 0

          for i, (img,cropped_img, coord, evs) in enumerate(train_loader):
               
               images = img.to(device)
               coord = coord.to(device)
               output_ball_detection_gs = model.ball_detection_GS(images)
               loss = criterion(output_ball_detection_gs, coord)

               optimizer.zero_grad()
               loss.backward()
               optimizer.step()

               predictions = output_ball_detection_gs.detach().argmax(1).to(device='cpu').numpy()[0]
               event = coord.to(device='cpu').numpy()
               gt.append(event)
               ed_output.append(predictions)
               correct_samples += ((predictions==event).sum())
               e_loss+=loss.item()
               
               if i == int(len/5):    # print every 50 mini-batches
                    print('Epoch: [%d][%d/%d] loss: %f' % \
                         (epoch + 1, i + 1, int(len/5) + 1, running_loss / 1))
               
               running_loss = 0.0

          accruracy_epoch.append(correct_samples/170*100)
          loss_epoch.append(e_loss/170)
          print('Loss:[%f], Correct Predictions[%d], Accuracy[%f]' % (e_loss/170, correct_samples,\
                correct_samples/170*100))

          if epoch%5==0:
                    # need to save model
               torch.save(model,f"checkpoints_gs\checkpoint_{epoch}.pth")
          

     return gt, ed_output, accruracy_epoch, loss_epoch





def train_model(
          device, model, train_loader,
          criterion, optimizer, epochs, 
          length):

     """
     this function will train the model with the following strategy, i.e., 
     - freeze the global_segment model, and train the local_segment block and the event spotting
       block.
     """

     epoch_loss_ball_detection = 0
     epoch_loss_event_detection = 0
     epoch_acc_event_detection = 0
     arr_loss_ball_detection = list()
     arr_loss_event_detection = list()
     accuracy_event_detection = list()


     for epoch in range(epochs):

          loss_ball_detection = 0
          loss_event_spotting = 0
          accuracy_event_spotting = 0

          loader = tqdm(train_loader)
          for i, (t_img, o_img, c_img, coordinates,event) in enumerate(loader):

               dg_images = t_img.to(device)
               label_coords = coordinates.to(device)
               label_events = event.to(device)
               

               # i don't need the output of this block as such
               # i would be backward propagating it with the loss of
               # event detection model
               gs_pred, global_features = model.ball_gs(dg_images)
               
               ls_pred, local_features = model.ball_ls(c_img)
               # coordinates converted to numpy and then perform formula
               # ball_coordinates = coordinates_constructor(
               #                          gs_pred.detach().numpy(), 
               #                          ls_pred.detach().numpy())

               loss_gs = criterion(gs_pred, label_coords)
               loss_ls = criterion(ls_pred, label_coords)
               loss_bd = loss_gs+loss_ls
               loss_ball_detection+=loss_bd.item()
               # now i have coordinates in their normalised form
               # time for applying loss function and loss propagation. 
               ed_output = model(local_features, global_features)
               loss_ed = criterion(ed_output, label_events)
               loss_event_spotting += loss_ed.item()
               acc = (ed_output.argmax(1) == label_events).type(torch.float).sum().item()/label_events.size(0)
               accuracy_event_spotting +=acc
               loss_bd.backward(retain_graph=True)
               optimizer.zero_grad()
               loss_ed.backward()
               optimizer.step()
               loader.set_description(f"Epoch[{epoch+1}/{epochs}],Step[{i+1}/{length}]")
               loader.set_postfix(loss=f"Ball detection: {loss_bd.item()}, Event Detection: {loss_ed.item()}",acc=acc)


          # loss for ball detection needs to be stored.
          # loss for event detection also needs to be stored.


          epoch_loss_ball_detection = loss_ball_detection/length
          epoch_loss_event_detection = loss_event_spotting/length
          epoch_acc_event_detection = accuracy_event_spotting/length

          arr_loss_ball_detection.append(epoch_loss_ball_detection)
          arr_loss_event_detection.append(epoch_loss_event_detection)
          accuracy_event_detection.append(epoch_acc_event_detection)
               

     return model, arr_loss_ball_detection, arr_loss_event_detection, accuracy_event_detection



     

def test(model, device, criterion, data):

     epoch=1
     correct = 0
     running_loss = 0
     len = 62
     ed_output = list()
     gt = list()
     for i, (img, cropped_img, coord, evs) in enumerate(data):

          images = img.to(device)
          crp_img = cropped_img.to(device)
          events_mrkup = evs.to(device)
          coord_ = coord.to(device)

          output_ball_detection_gs = model.ball_detection_GS(images)
          output_ball_detection_ls = model.ball_detection_LS(resize_around_coords(output_ball_detection_gs))
          output_event_spotting = model.event_spotting()
          # criterion for each block will be defined here
          # 

          loss = criterion(output_event_spotting, evs)
          predictions = output_event_spotting.detach().argmax(1).to(device='cpu').numpy()[0]
          event = evs.to(device='cpu').numpy()
          gt.append(evs.item())
          ed_output.append(predictions)
          correct += (( predictions==event).sum())
          running_loss += loss.item()
          if i == int(len/5):    # print every 50 mini-batches
               print('Epoch: [%d][%d/%d] loss: %f' % \
                    (epoch + 1, i + 1, int(len/5) + 1, running_loss / 1))
          
     print('Loss:[%f], Correct Predictions[%d], Accuracy[%f]' % (running_loss, correct, correct/62*100))
     return gt, ed_output




def main():
     
     choice = int(input("1. training from scratch\n2. resume training from\
           last checkpoint\n3. test model\n"))
     
     choices = {1:"training",2:"resume",3:"test"}
     model = EventSpotting()
     transform1 = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((320,128)), 
                                        transforms.Normalize(0,0.5)])
     transform2 = transforms.Compose([transforms.ToTensor()])
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

     criterion1 = nn.CrossEntropyLoss()
     learning_rate = 0.001
     optimizer = optim.Adam(model.parameters(),lr=learning_rate)


     root_dir = "dataset/root/"

     # what we want to do, training, testing or validation, i need to
     # to split data
     data_set = TTDataset(
                    root_dir, transform1=transform1, transform2=transform2, events)

     train_ds, test_ds = random_split(data_set, (170,62))

     train_loader = torch.utils.data.DataLoader(
                         dataset=train_ds, batch_size=5, shuffle=True)
     test_loader = torch.utils.data.DataLoader(
                         dataset=test_ds, batch_size=5, shuffle=True)

     if choices[choice] =="training":
          # create model
          # data loader
          # calling train function
          print("Training the model:   ")

          #     device,
     # model,
     # train_loader,
     # criterion, 
     # optimizer, 
     # epochs, 
     # length
     #
          model, acc, loss  = train_model(
                                   device=device,
                                   model=model,
                                   train_loader=train_loader,
                                   criterion=criterion1,
                                   optimizer=optimizer,
                                   epochs=100,
                                   length=len(train_ds))
          
          plt.plot(loss,"g")
          plt.plot(acc,"r")
          plt.show()
     elif choices[choice]=="resume":
          print("resume")
          
          model = torch.load("checkpoints_gs/checkpoint_50.pth",map_location=device)
          gt, ed_output, accruracy_epoch, loss_epoch = train_gs_block(device,model,train_loader,criterion1, \
               optimizer, epochs=30,len=len(train_ds))

     elif choices[choice]=="test":
          
          model = torch.load("checkpoints_gs/checkpoint_100.pth",map_location=device)
          test(model,device,test_loader,criterion1)

     else:
          error("Invalid arguments")


     model = torch.load("checkpoints/checkpoint_10.pth",map_location=device)
     model._req_grad(5,False)
     gt, event= test(model, device, criterion1, test_loader)
     confusion_matrix_plot(event, gt)



if __name__=='__main__':
     main()

