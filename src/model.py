import torch
import pandas as pd 
import numpy as np 
import copy 
from tqdm import tqdm
import pickle 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import torch.nn.functional as F

# SEED = 42 
# np.random.seed(SEED)
# random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Receiver operating characteristic curve plots false positive rate versus true positive rate, so captures only recall. 
# Essentially gives "what is the probability that a randomly-chosen positive example is ranked more highly than a 
# randomly-chosen negative example?"
# TPR is (true positives) / (true positives + false negatives), which is the same as recall. 
# FPR is (false positives / (false positives + true negatives)), "out of all true negatives, how many were messed up?"
# Precision is (true positives) / (true positives + false positives)
def get_roc_auc(labels, outputs):
    '''Compute the ROC AUC'''
    return float(roc_auc_score(labels, outputs, average='macro'))

def get_pr_auc(labels, outputs):
    '''Approximates the area under the precision recall curve.'''
    return float(average_precision_score(labels, outputs))

def get_metrics(labels, outputs):
    metrics = dict()
    metrics['roc_auc_1'] = get_roc_auc(labels, outputs)
    metrics['roc_auc_0'] = get_roc_auc(1 - labels, 1 - outputs)
    metrics['pr_auc_1'] = get_pr_auc(labels, outputs)
    metrics['pr_auc_0'] = get_pr_auc(1 - labels, 1 - outputs)
    return metrics


class MLP(torch.nn.Module):

    def __init__(self, model_id:int=0):

        super(MLP, self).__init__()
    
        self.model_id = f'mlp_{model_id}' 

        layers = [torch.nn.Linear(1280, 512, dtype=torch.float32), torch.nn.ReLU()]
        layers += [torch.nn.Linear(512, 256, dtype=torch.float32), torch.nn.ReLU()]
        layers += [torch.nn.Linear(256, 128, dtype=torch.float32), torch.nn.ReLU()]
        layers += [torch.nn.Linear(128, 1, dtype=torch.float32)]
        self.model = torch.nn.Sequential(*layers) # Initialize the sequential model. 
        self.scaler = StandardScaler()
        self.to(DEVICE)

    
    def save(self, path:str=None):
        info = {'model_id':self.model_id}
        info['scaler'] = self.scaler
        # Pickling the full model can break if the class definition changes or Python and PyTorch versions differ.
        # Just want to save the state dict for compatibility.
        info['model'] = self.state_dict()
        if path is None:
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
        embeddings_train, embeddings_test = dataset_train.to_numpy()[0], dataset_test.to_numpy()[0]
        dataset_train.embeddings = torch.FloatTensor(self.scaler.fit_transform(embeddings_train))
        dataset_test.embeddings = torch.FloatTensor(self.scaler.transform(embeddings_test))
        return dataset_train, dataset_test
    
    def _get_loss(self, outputs, targets):
        '''Implement weighted cross-entropy loss.'''
        # F.cross_entropy applies sigmoid under the hood, so do not apply twice.
        # return F.cross_entropy(outputs, targets, weight=self.loss_weights)
        weights = torch.where(targets == 1, 1, self.loss_weights[0])
        return F.binary_cross_entropy_with_logits(outputs, targets, weight=weights)

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
            
            test_losses.append(float(test_loss)) # Make sure these are floats so they are json-serializable.
            train_losses.append(float(np.mean(train_loss)))
        
        self.load_state_dict(best_model_weights)
        return {'train_loss':train_losses, 'test_loss':test_losses, 'epoch':list(range(epochs))}

    def predict(self, dataset) -> pd.DataFrame:

        embeddings, labels = dataset.to_numpy()
        embeddings = self.scaler.transform(embeddings)
        embeddings = torch.FloatTensor(embeddings).to(DEVICE)

        self.eval() # Put the model in evaluation mode. This changes the forward behavior of the model (e.g. disables dropout).
        with torch.no_grad(): # Turn off gradient computation, which reduces memory usage. 
            outputs = self(embeddings) # Run a forward pass of the model.
            outputs = torch.nn.functional.sigmoid(outputs) # Apply sigmoid activation, which is applied as a part of the loss function during training. 
            outputs = outputs.cpu().numpy().ravel()
        
        result = {'outputs':outputs.tolist()}
        if labels is not None:
            result.update(get_metrics(labels, outputs))
            result['labels'] = labels.tolist()

        return result


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
        if path is None:
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
    



