import torch
import pandas as pd 
import numpy as np 
import copy 
from tqdm import tqdm
import pickle 
from sklearn.preprocessing import StandardScaler

# SEED = 42 
# np.random.seed(SEED)
# random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)

def get_auroc():
    pass 



class MLP(torch.nn.Module):

    def __init__(self, model_id:int=0):

        super(MLP, self).__init__()
    
        self.model_id = f'mlp_{model_id}' 

        layers = [torch.nn.Linear(1024, 512, dtype=torch.float32), torch.nn.ReLU()]
        layers += [torch.nn.Linear(512, 256, dtype=torch.float32), torch.nn.ReLU()]
        layers += [torch.nn.Linear(256, 128, dtype=torch.float32), torch.nn.ReLU()]
        layers += [torch.nn.Linear(128, 2, dtype=torch.float32)]
        self.model = torch.nn.Sequential(*layers) # Initialize the sequential model. 
        self.scaler = StandardScaler()
        self.to(DEVICE)

    def _fit_loss_weights(self, dataset_train, alpha:float=0.5):
        n_1, n_0 = (dataset_train.labels == 1).sum(), (dataset_train.labels == 0).sum()
        assert n_1 > n_0, 'MLP.loss: Expect label 1 to be the majority class.'
        self.loss_weights = [(n_1 / n_0)**alpha, 1]
        self.loss_weights = torch.FloatTensor(self.loss_weights).to(DEVICE)

    def _fit_scaler(self, dataset_train, dataset_test):
        self.scaler.fit(dataset_train.embeddings)
        dataset_train.embeddings = self.scaler.transform(dataset_train.embeddings)
        dataset_test.embeddings = self.scaler.transform(dataset_test.embeddings)
        return dataset_train, dataset_test
    
    def _get_loss(self, outputs, targets):
        '''Implement weighted cross-entropy loss.'''
        # F.cross_entropy applies softmax under the hood, so do not apply twice.
        return F.cross_entropy(outputs, targets, weight=self.loss_weights)

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

        best_epoch, best_test_loss = 0, np.inf
        best_model_weights = copy.deepcopy(self.state_dict())
        test_losses, train_losses = list(), list()

        for epoch in tqdm(range(epochs), desc=f'MLP.fit: Training model {self.model_id}.'):
            train_loss = list() # Re-initialize the epoch loss. 
            for batch in loader:
                outputs, targets = self(batch['embedding']), batch['label']
                # outputs, targets = self(batch['embedding']).to(DEVICE), batch['label'].to(DEVICE).long()
                loss = self._get_loss(outputs, targets)
                # Gradients accumulate by default, so if you don’t zero them before computing new gradients, you silently use the wrong gradient.
                # Gradients must be zero before calling loss.backwards. 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                train_loss.append(loss.item()) # Store each loss for computing metrics.  

            test_inputs, test_targets = dataset_test.embeddings, dataset_test.labels
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


class Tree():
    pass 



class Ensemble():


    def __init__(self, n_models:int=5, model_class=MLP):

        self.n_models = n_models
        self.models = [model_class(model_id=i) for i in range(n_models)]

    def fit(self, datasets, epochs:int=100, lr:float=1e-4, batch_size:int=64, alpha:float=0.5):

        loss_df = list()
        for model in self.models:
            losses = model.fit(datasets, epochs=epochs, lr=lr, batch_size=batch_size, alpha=alpha)
            loss_df.append(pd.DataFrame(losses).assign(model_id=model.model_id))
        loss_df = pd.concat(loss_df)

        return loss_df

    def save(self, path):
        info = {'n_models': self.n_models}
        info['model_ids'] = [m.model_id for m in self.models]
        # Pickling the full model can break if the class definition changes or Python and PyTorch versions differ.
        # Just want to save the state dict for compatibility.
        info['state_dicts'] = [m.state_dict() for m in self.models],
        info['scalers'] = [m.scaler for m in self.models]
        with open(path, 'wb') as f:
            pickle.dump(info, f)

    @classmethod
    def load(cls, path, model_class=MLP):
        with open(path, 'rb') as f:
            info = pickle.load(f)
        
        obj = cls(n_models=info['n_models'], model=model_class)
        for model, state_dict, scaler in zip(obj.models, info['state_dicts'], info['scalers']):
            model.load_state_dict(state_dict)
            model.scaler = scaler
        
        return obj



