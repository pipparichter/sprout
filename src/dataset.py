import pandas as pd
import numpy as np
import torch
from torch.nn.functional import one_hot
import copy
import tables 
from tqdm import tqdm
import warnings 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Dataset(torch.utils.data.Dataset):
    label_map = {'spurious':0, 'real':1}

    def __init__(self, embeddings:np.ndarray=None, index:np.ndarray=None, labels:np.ndarray=None, metadata:pd.DataFrame=None):

        self.path = path # Store the path from which the Dataset was loaded. 

        self.metadata = metadata
        self.index = index
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE) if (embedding is not None) else embedding
        self.labels = torch.tensor(labels, dtype=torch.long).to(DEVICE) if (labels is not None) else None

    def __len__(self):
        return len(self.index)
    

    @classmethod
    def from_hdf(cls, path:str):
        embedding_df = pd.read_hdf(path, key='embeddings')
        index = embedding_df.index.values.copy() # Make sure this is a numpy array. 
        embeddings = embedding_df.values.copy() # Why do I need to copy this?

        try:
            metadata = Dataset._read_hdf(path, key='metadata')
            assert len(embedding_df) == len(metadata_df), 'Dataset.from_hdf: The indices of the embedding and the metadata do not match.'
            assert np.all(embedding_df.index == metadata_df.index), 'Dataset.from_hdf: The indices of the embedding and the metadata do not match.'
        except Exception: 
            print(f'Dataset.from_hdf: No metadata stored in the Dataset')

        return cls(embeddings, index=index, metadata=metadata)
    

    def __getitem__(self, idx:int) -> dict:
        item = {'embedding':self.embeddings[idx]} 
        if (self.labels is not None):
            item['label'] = self.labels[idx]
        return item