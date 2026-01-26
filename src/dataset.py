import pandas as pd
import numpy as np
import torch
import copy
from tqdm import tqdm
import warnings 
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Dataset(torch.utils.data.Dataset):
    label_map = {'spurious':0, 'real':1}

    def __init__(self, embeddings:np.ndarray=None, index:np.ndarray=None, labels:np.ndarray=None, metadata:pd.DataFrame=None):

        self.metadata = metadata
        self.index = index
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE) if (embeddings is not None) else embeddings
        self.labels = torch.tensor(labels, dtype=torch.long).to(DEVICE) if (labels is not None) else None

    def __len__(self):
        return len(self.index)
    
    def to_numpy(self, labels:bool=False):
        '''Convert embeddings and labels to a numpy array.'''
        embeddings = self.embeddings.cpu().numpy()
        if labels:
            labels = self.labels.cpu().numpy()
            return embeddings, labels 
        else:
            return embeddings
        
    def subset(self, idxs:np.ndarray):
        embeddings = copy.deepcopy(self.embeddings[idxs, :])
        index = self.index[idxs].copy()
        labels = copy.deepcopy(self.labels) if (self.labels is not None) else self.labels
        metadata = self.metadata.iloc[idxs]
        return Dataset(embeddings=embeddings, index=index, labels=labels, metadata=metadata)
    
    @classmethod
    def from_hdf(cls, path:str):
        embedding_df = pd.read_hdf(path, key='embeddings')
        index = embedding_df.index.values.copy() # Make sure this is a numpy array. 
        embeddings = embedding_df.values.copy() # Why do I need to copy this?
        metadata, labels = None, None
        try:
            metadata = pd.read_hdf(path, key='metadata')
            labels = (metadata.label.values) if ('label' in metadata.columns) else None
            assert len(embedding_df) == len(metadata), 'Dataset.from_hdf: The indices of the embedding and the metadata do not match.'
            assert np.all(embedding_df.index == metadata.index), 'Dataset.from_hdf: The indices of the embedding and the metadata do not match.'
        except Exception: 
            print(f'Dataset.from_hdf: No metadata stored in the Dataset')

        return cls(embeddings, index=index, metadata=metadata, labels=labels)
    

    def __getitem__(self, idx:int) -> dict:
        item = {'embedding':self.embeddings[idx]} 
        if (self.labels is not None):
            item['label'] = self.labels[idx]
        return item
    


def split(dataset, random_state:int=42):
    '''Split the input dataset into two parts for training and validation.'''

    _, labels = dataset.to_numpy(labels=True)
    idxs = np.arange(len(dataset))
    train_idxs, test_idxs = train_test_split(idxs, test_size=0.2, stratify=labels, random_state=random_state)

    dataset_train = dataset.subset(train_idxs)
    dataset_test = dataset.subset(test_idxs)
    return dataset_train, dataset_test

