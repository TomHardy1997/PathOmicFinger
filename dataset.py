import torch
import ipdb
import pandas as pd
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np
import os

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import torch.utils.data
import ast

'''
This Dataset is designed to predict adverse drug reactions 
and it incorporates genetic information and drug information
'''
class Response_WSI_Gene_Dataset(Dataset):
    def __init__(self, csv_path, dict, data_dir, mode):
        super(Response_WSI_Gene_Dataset, self).__init__()
        df = pd.read_csv(csv_path, low_memory=False)
        df = df.sample(frac=1, random_state=1)
        dict = dict
        self.slide = df['slide_id']
        self.label_col = df['measure_of_response'].map(dict)
        self.data_dir = data_dir 
        self.num_classes = len(self.label_col.unique())
        self.case_id = df['case_id']
        self.drug = df['finger']
        self.gene = df.iloc[:, 13:]
        self.label = torch.tensor(self.label_col)
        self.idx0 = torch.where(self.label==0)[0]
        self.idx1 = torch.where(self.label==1)[0]
        self.mode = mode

    def __len__(self):
        return len(self.slide)


    def __getitem__(self, idx):
        if self.mode == 'pathfusion':
            slide_id = self.slide[idx]
            wsi = os.path.join(self.data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
            wsi_bag = torch.load(wsi)
            label = torch.tensor(self.label_col[idx])
            gene = torch.tensor(self.gene.iloc[idx], dtype=torch.float32)
            finger = torch.tensor(ast.literal_eval(self.drug[idx]))
            
            return slide_id, label, gene.unsqueeze(dim=0), wsi_bag, finger
        
        elif self.mode == 'path':
            slide_id = self.slide[idx]
            wsi = os.path.join(self.data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
            wsi_bag = torch.load(wsi)
            label = torch.tensor(self.label_col[idx])
            
            return slide_id, label, wsi_bag
        
        elif self.mode == 'gene':
            slide_id = self.slide[idx]
            label = torch.tensor(self.label_col[idx])
            gene = torch.tensor(self.gene.iloc[idx], dtype=torch.float32)

            return slide_id, label, gene.unsqueeze(dim=0)
        
        elif self.mode == 'pathgene':
            slide_id = self.slide[idx]
            wsi = os.path.join(self.data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
            wsi_bag = torch.load(wsi)   
            label = torch.tensor(self.label_col[idx])
            gene = torch.tensor(self.gene.iloc[idx], dtype=torch.float32)
            
            return slide_id, label, gene.unsqueeze(dim=0), wsi_bag
        
        elif self.mode == 'pathfinger':
            slide_id = self.slide[idx]
            wsi = os.path.join(self.data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
            wsi_bag = torch.load(wsi)
            label = torch.tensor(self.label_col[idx])
            finger = torch.tensor(ast.literal_eval(self.drug[idx]))
            
            return slide_id, label, finger, wsi_bag
        
        elif self.mode == 'genefinger':
            slide_id = self.slide[idx]
            gene = torch.tensor(self.gene.iloc[idx], dtype=torch.float32)
            label = torch.tensor(self.label_col[idx])   
            finger = torch.tensor(ast.literal_eval(self.drug[idx]))

            return slide_id, label, gene.unsqueeze(dim=0), finger


def custom_collate_fn(batch):
    slide_ids, labels, genes, path_features, fingers = zip(*batch)
    path_features = pad_sequence(path_features, batch_first=True)

    return slide_ids, labels, genes, path_features, fingers

if __name__ == '__main__':
    csv_path = '/home/stat-jijianxin/gene/PORPOISE-master/datasets_csv/tcga_gbmlgg_trian_new_finger_clean.csv.zip'
    data_dir = '/home/stat-jijianxin/gene/PORPOISE-master/TCGA-GBM/'
    dict = {'Stable Disease': 0, 'Partial Response': 1, 'Complete Response': 1, 'Clinical Progressive Disease':0}
    dataset = Response_WSI_Gene_Dataset(csv_path = csv_path, data_dir = data_dir, dict = dict, mode = 'genefinger')
    # import ipdb;ipdb.set_trace()
    class_counts = [len(dataset.idx0), len(dataset.idx1)]
    class_weights = 1.0 / torch.Tensor(class_counts)
    sampler = WeightedRandomSampler(class_weights, len(dataset), replacement=True)
    train_loader = DataLoader(dataset, batch_size=12, num_workers=0, collate_fn=custom_collate_fn,sampler=sampler, shuffle=False)
    import ipdb; ipdb.set_trace()