from PIL import Image
import torch
import pandas as pd
from  torchvision import transforms
from torch.utils.data import Dataset, DataLoader  

def label_tag(img_label):
    
    ref={'Benign':0, 'Stroma':1, 'Tumor':2, 'Debris':3, 'Inflammatory':4, 'Muscle':5}
    return ref[img_label]

class FhistDataset:
    #Load the Dataset
    def __init__(self,transforms=transforms.ToTensor(),path='/home/ravi/Domain_adap_code/data_source_train.csv'):
        # store the inputs and outputs
        self.path=path
        self.transforms=transforms
        self.df=pd.read_csv(self.path)

    def __len__(self):
        return len(self.df)

    # get a row at an index
    def __getitem__(self, idx):
        one_row=self.df.loc[idx,'path']
        img=Image.open(one_row)
        #print(img)
        # label=onerow.split('/')[-2]
        label=torch.tensor(self.df.loc[idx,'label'])            #use label_tag(self.df.loc[idx,'label']) if ypur csv is not having numeric for class
        #print(label)
        if self.transforms is not None:
            img=self.transforms(img)
        return img ,label#,one_row



########## below code is for mean and std for data

batch_size=512
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = FhistDataset(path='/home/ravi/Domain_adap_code/data_source_train.csv', transforms=transform)
source_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
def get_mean_std(loader):
  channels_sum, channels_square_sum, num_batches = 0, 0, 0
  for data, _ in loader:
    channels_sum += torch.mean(data.type(torch.FloatTensor), dim = [0,2,3])
    channels_square_sum += torch.mean(data.type(torch.FloatTensor)**2, dim = [0,2,3])
    num_batches += 1

  mean = channels_sum/num_batches
  std = (channels_square_sum/num_batches - mean**2)**0.5
  return mean, std
mean, std = get_mean_std(source_loader)
print(mean, std)     
