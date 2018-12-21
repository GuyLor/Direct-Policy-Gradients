
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import saved_models
import os

class Policy(nn.Module):
    """ policy for gym-minigrid env """
    def __init__(self, action_space):
        super(Policy, self).__init__()
        
        self.image_embedding_size = 64
        self.number_directions = 4
        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3))           
        )
        self.fc = nn.Sequential(
            
            nn.Linear(self.image_embedding_size+self.number_directions, action_space)

        )

    def forward(self,batch_obs):
        images = []
        directions = []
        for obs in batch_obs:
            image = np.array([obs["image"]])
            image = torch.tensor(image,dtype=torch.float)
            x = torch.transpose(torch.transpose(image, 1, 3), 2, 3)
            images.append(x)
            
            direction = torch.LongTensor([obs["direction"]]).unsqueeze(0)
            direction = torch.FloatTensor(direction.size(0),self.number_directions).zero_().scatter_(-1, direction,1.0)
            directions.append(direction)
        x = torch.cat(images)
        direction = torch.cat(directions)
        x = self.image_conv(x)
        x = x.view(x.size(0), -1)
        image_direction = torch.cat((x,direction),dim=1)
        logits = self.fc(image_direction)
        return F.log_softmax(logits,dim=-1)

class Checkpoint:
    def __init__(self,folder_path,load_path,save_path):
        if folder_path is not None:
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
            self.folder_path = folder_path
        self.load_path = load_path
        self.save_path = save_path        
        
    def save(self,model,optimizer):
        if self.save_path is not None:
            state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
            }
            filepath=os.path.join(self.folder_path,self.save_path)
            torch.save(state, filepath)

    def load(self,model,optimizer):
        if self.load_path is not None:
            # "lambda" allows to load the model on cpu in case it is saved on gpu
            filepath=os.path.join(self.folder_path,self.load_path)
            state = torch.load(filepath,lambda storage, loc: storage)
            model.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            return model,optimizer

