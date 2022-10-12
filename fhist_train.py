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
        return img,label,one_row






     
