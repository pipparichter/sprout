import torch
import pandas as pd 
import numpy as np 
import copy 
from tqdm import tqdm
import pickle 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import torch.nn.functional as F

# SEED = 42 
# np.random.seed(SEED)
# random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_auroc(labels, outputs):
    '''Compute the ROC AUC (Area Under the Receiver Operating Characteristic Curve) score, which evaluates model 
    performance in a threshold-agnostic way. It is based on the probability of a positive example being ranked more highly
    than a negative example.'''
    return roc_auc_score(labels, outputs, average='macro') 


class MLP(torch.nn.Module):

    def __init__(self, model_id:int=0):

        super(MLP, self).__init__()
    
        self.model_id = f'mlp_{model_id}' 

        layers = [torch.nn.Linear(1280, 512, dtype=torch.float32), torch.nn.ReLU()]
        layers += [torch.nn.Linear(512, 256, dtype=torch.float32), torch.nn.ReLU()]
        layers += [torch.nn.Linear(256, 128, dtype=torch.float32), torch.nn.ReLU()]
        layers += [torch.nn.Linear(128, 2, dtype=torch.float32)]
        self.model = torch.nn.Sequential(*layers) # Initialize the sequential model. 
        self.scaler = StandardScaler()
        self.to(DEVICE)

    
    def save(self, path:str=None):
        info = {'model_id':self.model_id}
        info['scaler'] = self.scaler
        # Pickling the full model can break if the class definition changes or Python and PyTorch versions differ.
        # Just want to save the state dict for compatibility.
        info['model'] = self.state_dict()
        if path is not None:
            return info
        with open(path, 'wb') as f:
            pickle.dump(info, f)

    
    @classmethod
    def from_dict(cls, info:dict):
        obj = cls()
        obj.model_id = info.get('model_id', '')
        obj.load_state_dict(info['model'])
        obj.scaler = info['scaler']
        return obj
    
    @classmethod
    def load(path:str):
        with open(path, 'rb') as f:
            info = pickle.load(f)
        return MLP.from_dict(info)


    def _fit_loss_weights(self, dataset_train, alpha:float=0.5):
        n_1, n_0 = (dataset_train.labels == 1).sum(), (dataset_train.labels == 0).sum()
        assert n_1 > n_0, 'MLP.loss: Expect label 1 to be the majority class.'
        self.loss_weights = [(n_1 / n_0)**alpha, 1]
        self.loss_weights = torch.FloatTensor(self.loss_weights).to(DEVICE)

    def _fit_scaler(self, dataset_train, dataset_test):
        self.scaler.fit(dataset_train.to_numpy())
        dataset_train.embeddings = torch.FloatTensor(self.scaler.transform(dataset_train.to_numpy()))
        dataset_test.embeddings = torch.FloatTensor(self.scaler.transform(dataset_test.to_numpy()))
        return dataset_train, dataset_test
    
    def _get_loss(self, outputs, targets):
        '''Implement weighted cross-entropy loss.'''
        # F.cross_entropy applies softmax under the hood, so do not apply twice.
        return F.cross_entropy(outputs, targets, weight=self.loss_weights)

    @staticmethod
    def _get_dataloader(dataset_train, alpha:float=0, batch_size:int=64):
        n_1, n_0 = (dataset_train.labels == 1).sum(), (dataset_train.labels == 0).sum()
        assert n_1 > n_0, 'MLP.loss: Expect label 0 to be the minority class.'
        weights = [(n_1 / n_0)**alpha, 1]
        weights = [weights[label] for label in dataset_train.labels]
        sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(dataset_train), replacement=True)
        loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=sampler)
        return loader 

    def forward(self, inputs:torch.FloatTensor):
        return self.model(inputs)

    def fit(self, datasets, epochs:int=100, lr:float=1e-4, batch_size:int=64, alpha:float=0.5):

        self.train() # Put the model in train mode.
        
        dataset_train, dataset_test = datasets
        dataset_train, dataset_test = self._fit_scaler(dataset_train, dataset_test)
        self._fit_loss_weights(dataset_train, alpha=alpha) # Set the weights of the loss function.
        optimizer = torch.optim.Adam(self.parameters(), lr=lr) # Optimizer updates model parameters based on loss gradients.
        loader = self._get_dataloader(dataset_train, alpha=0)

        best_test_loss = np.inf
        best_model_weights = copy.deepcopy(self.state_dict())
        test_losses, train_losses = list(), list()

        for epoch in tqdm(range(epochs), desc=f'MLP.fit: Training model {self.model_id}.'):
            train_loss = list() # Re-initialize the epoch loss. 
            for batch in loader:
                outputs, targets = self(batch['embedding'].to(DEVICE)), batch['label'].to(DEVICE)
                # outputs, targets = self(batch['embedding']).to(DEVICE), batch['label'].to(DEVICE).long()
                loss = self._get_loss(outputs, targets)
                # Gradients accumulate by default, so if you don’t zero them before computing new gradients, you silently use the wrong gradient.
                # Gradients must be zero before calling loss.backwards. 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                train_loss.append(loss.item()) # Store each loss for computing metrics.  

            test_inputs, test_targets = dataset_test.embeddings.to(DEVICE), dataset_test.labels.to(DEVICE)
            test_outputs = self(test_inputs)
            test_loss = self._get_loss(test_outputs, test_targets).item()

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_weights = copy.deepcopy(self.state_dict())
                print(f'MLP.fit: New best model weights for {self.model_id} found after epoch {epoch}. Test loss is {test_loss}.', flush=True)
            
            test_losses.append(test_loss)
            train_losses.append(np.mean(train_loss))
        
        self.load_state_dict(best_model_weights)
        return {'train_loss':train_losses, 'test_loss':test_losses, 'epoch':np.arange(epochs)}

    def predict(self, dataset, scale:bool=False) -> pd.DataFrame:

        self.eval() # Put the model in evaluation mode. This changes the forward behavior of the model (e.g. disables dropout).
        with torch.no_grad(): # Turn off gradient computation, which reduces memory usage. 
            outputs = self(dataset.embeddings) # Run a forward pass of the model.
            outputs = torch.nn.functional.softmax(outputs, 1) # Apply sigmoid activation, which is applied as a part of the loss function during training. 
            outputs = outputs.cpu().numpy()
        
        return {'model_output_0':outputs[:, 0].ravel(), 'model_output_1':outputs[:, 1].ravel()}


# https://xgboost.readthedocs.io/en/stable/python/python_intro.html
class Tree():

    # params = {'max_depth':2, 'eta':1, 'objective': 'binary:logistic'}
    # params['nthread'] = 4
    # params['eval_metric'] = 'auc'

    @staticmethod
    def _get_dmatrix(dataset):

        embeddings, labels = dataset.to_numpy(labels=True)
        dataset = xgb.DMatrix(embeddings, label=labels)
        return dataset 
    

    def save(self, path:str=None):
        info = {'model_id':self.model_id}
        info['scaler'] = self.scaler
        info['model'] = self.model.save_raw() # Serializes the tree. 
        if path is not None:
            return info
        with open(path, 'wb') as f:
            pickle.dump(info, f)

    @classmethod
    def from_dict(cls, info:dict):
        obj = cls()
        obj.model_id = info.get('model_id', '')
        obj.model = xgb.Booster()
        obj.model.load_model(info['model'])
        obj.scaler = info['scaler']
        return obj
    
    @classmethod
    def load(path:str):
        with open(path, 'rb') as f:
            info = pickle.load(f)
        return Tree.from_dict(info)
    
    @classmethod
    def load(cls, model, scaler, model_id:int=''):
        obj = cls()
        obj.scaler = scaler 
        obj.model_id = model_id
        obj.model = xgb.Booster()
        obj.model.load_model(model)
        return obj
    
    def _fit_scaler(self, dataset_train:np.ndarray, dataset_test:np.ndarray):
        self.scaler.fit(dataset_train)
        dataset_train = self.scaler.transform(dataset_train)
        dataset_test = self.scaler.transform(dataset_test)
        return dataset_train, dataset_test
    
    def __init__(self, model_id:int=0):

        self.model_id = f'tree_{model_id}' 
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, datasets, num_rounds, **params):
        dataset_train, dataset_test = datasets
        dataset_train, dataset_test = self._fit_scaler(dataset_train, dataset_test)
        dataset_train = Tree._get_dmatrix(dataset_train)
        dataset_test = Tree._get_dmatrix(dataset_test)
        eval_list = [(dataset_train, 'train'), (dataset_test, 'eval')]

        self.model = xgb.train(params, dataset_train, num_rounds, eval_list)
        self.model.set_attr(model_id=self.model_id)

    def predict(self, dataset):
        pass



class Ensemble():

    def __init__(self, n_models:int=5, model_class=MLP, init_models:bool=True):

        self.n_models = n_models
        self.models = [] if (not init_models) else [model_class(model_id=i) for i in range(n_models)]

    def fit(self, datasets, epochs:int=100, lr:float=1e-4, batch_size:int=64, alpha:float=0.5):

        loss_df = list()
        for model in self.models:
            losses = model.fit(datasets, epochs=epochs, lr=lr, batch_size=batch_size, alpha=alpha)
            loss_df.append(pd.DataFrame(losses).assign(model_id=model.model_id))
        loss_df = pd.concat(loss_df)

        return loss_df

    def save(self, path):
        info = {'n_models': self.n_models}
        info['models'] = [model.save(path=None) for model in self.models]
        with open(path, 'wb') as f:
            pickle.dump(info, f)

    @classmethod
    def load(cls, path, model_class=MLP):
        with open(path, 'rb') as f:
            info = pickle.load(f)
        
        obj = cls(n_models=info['n_models'], model_class=model_class, init_models=False)
        obj.models = [model_class.from_dict(model_info) for model_info in info['models']]
        return obj
    



