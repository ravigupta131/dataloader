import os
import pandas as pd

top = '/home/ravi/Domain_adap_code/fhist_dataset/NCT_Target/Data_target_folder'

data_record = {'path':[], 'slide_name':[],'label':[]}
for root, directories, files in os.walk(top, topdown=False):
    for name in files:
        print(root)
        print(name)
        data_record['slide_name'].append(name)
        data_record['path'].append(os.path.join(root, name))
        data_record['label'].append(root.split('/')[-1])

df = pd.DataFrame(data_record)

df.label[df.label=='Benign']=0
df.label[df.label=='Stroma']=1
df.label[df.label=='Tumor']=2
df.label[df.label=='Debris']=3
df.label[df.label=='Inflammatory']=4
df.label[df.label=='Muscle']=5


df.to_csv('data_target_test.csv',index=False)

        