import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data 
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # param [input_c, output_c, kernel_size, stride, padding]
            nn.Conv2d(2, 64, 7, 2, 3),   # [-1, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            # nn.Tanh(),
            
            nn.Conv2d(64, 64, 7, 2, 3),  # [-1, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            # nn.ReLU(),
            
            nn.Conv2d(64, 64, 5, 2, 2),  # [-1, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            # nn.Tanh(),
            
            nn.Conv2d(64, 128, 5, 2, 2),  #  [-1, 128, 8, 8]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            # nn.ReLU(),
            
            nn.Conv2d(128, 128, 5, 2, 2), # [, 64, 4, 4]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            # nn.Tanh(),
            
            nn.Conv2d(128, 128, 2, 2), # [, 64, 4, 4]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            # nn.ReLU(),

            nn.Conv2d(128, 64, 2, 1), # [, 64, 4, 4]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            # nn.Tanh(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 2 ,1),   # [, 128, 24, 24]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            # nn.ReLU(),

            nn.ConvTranspose2d(128, 128, 2, 2),   # [, 128, 48, 48]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            # nn.Tanh(),

            nn.ConvTranspose2d(128, 128, 6, 2, 2),    # [, 64, 48, 48]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            # nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 6, 2, 2),    # [, 64, 48, 48]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            # nn.Tanh(),

            nn.ConvTranspose2d(64, 64, 6, 2, 2),      # [, 32, 48, 48]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            # nn.ReLU(),

            nn.ConvTranspose2d(64, 64, 8, 2, 3),      # [, 32, 48, 48]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            # nn.Tanh(),
            
            nn.ConvTranspose2d(64, 2, 8, 2, 3)
            
        )
    
    def forward(self, x):
        y = self.encoder(x)
        z = self.decoder(y)
        return y, z
    def encode(self,x):
        return self.encoder(x)


def mask(data):
    mask=np.zeros(data.shape)
    mask[:,:,4:124,4:124]=1
    data[mask==0]=0
    return data
   
def normalize(data, mean, std):
    for i in range(data.shape[1]):
        data[:,i,:,:] = (data[:,i,:,:] - mean[i]) / std[i]
    return data

def inv_normalize(data, mean, std):
    for i in range(data.shape[1]):
        data[:,i,:,:] = data[:,i,:,:] * std[i] + mean[i]
    return data
    
class GetData(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.len = self.dataset.shape[0]

    def __getitem__(self, idx):
        img = self.dataset[idx, :, :, :]
        return idx, img

    def __len__(self):
        return self.len


if __name__=='__main__':
    model=AutoEncoder()
    model=model.cuda()
    criterion=nn.MSELoss()
    lr=1e-3
    path='/home/shared/tube_DRL/a_2'
    epoches=100
    optimizer=optim.Adam(model.parameters(),lr=lr)
    # scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    files = os.listdir(path)
    train_dataset = []
    print('Start Collect Data')
    print(files[1000:1050])
    for file in tqdm(files):
        train_data=np.load(path+'/'+file)['data'].reshape([-1,2,128,128])
        # train_data=mask(train_data)
        train_dataset.append(train_data)
    train_dataset = np.array(train_dataset).reshape([-1,2,128,128])
    mean = np.zeros(train_dataset.shape[1])
    std = np.zeros(train_dataset.shape[1])
    for i in range(train_dataset.shape[1]):
        mean[i] = train_dataset[:,i,:,:].mean()
        std[i] = train_dataset[:,i,:,:].std()
    np.savez('field.npz', mean=mean, std=std)
    print(train_dataset.shape)
    print(mean)
    print(std)
    train_dataset = torch.tensor(train_dataset)
    train_dataset = +(train_dataset, mean, std)
    train_dataset = GetData(train_dataset)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True, num_workers=4)
    print('Finish Data')
    for epoch in tqdm(range(epoches)):
        train_loss_epoch=0
        train_batch_num=0

        for step, data in enumerate(train_loader):
            optimizer.zero_grad()
            _, img = data
            img=img.cuda()
            _, output = model(img)
            output = output.cuda()
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
            train_batch_num += 1
            
        print('epoch:{} loss:{:2f}'.format(epoch+1,train_loss_epoch/train_batch_num))
        # scheduler.step()

    model = model.cpu()    
    torch.save(model.state_dict(), "model/autoencoder_full.pth")

    model.cpu()
    test_dataset = []
    for file in tqdm(files[1000:2000]):
        test_data = np.load(path+'/'+file)['data'].reshape([-1,2,128,128])
        test_dataset.append(test_data)
    test_dataset = np.array(test_dataset).reshape([-1,2,128,128])
    test_dataset = torch.tensor(test_dataset)
    test_dataset = normalize(test_dataset, mean, std)
    _, output = model(test_dataset)
    output = output.detach().numpy()
    output = inv_normalize(output, mean, std)
    np.savez('output.npz', output=output)
