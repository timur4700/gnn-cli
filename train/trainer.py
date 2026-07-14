from typing import Literal, Tuple
from pathlib import Path

from torch.optim import Adam, AdamW
import torch
from torch.nn import MSELoss, Module
from torch_geometric.loader import DataLoader
from train.utils import split_data

from scipy.stats import pearsonr, spearmanr

from sklearn.metrics import r2_score, root_mean_squared_error as rmse, mean_absolute_error as mae

import numpy as np

def make_optimizer(optim: Literal['adam', 'adam_w']):

    if optim == 'adam':
        return Adam

    return AdamW


def make_loss_func(loss: Literal['mse']):

    if loss == 'mse':
        return MSELoss


def rp_calc(y_test: np.ndarray, y_hat: np.ndarray):
    return pearsonr(y_test, y_hat)[0]


def rs_calc(y_test: np.ndarray, y_hat: np.ndarray):
    return spearmanr(y_test, y_hat)[0]



class Trainer():
    def __init__(self,
                 optimizer: Literal['adam', 'adam_w'],
                 seed: int,
                 n_epochs: int,
                 learning_rate: float,
                 weight_dacay: float=0,
                 early_stop: bool=False,
                 when_early_stop: int=100,
                 loss_func: Literal['mse']='mse',
                 verbose: Literal[0, 1]=1,
                 device: str='cpu',
                 save_train_log: bool=False
                 ):

        self.device = device

        self.seed = seed
        self.n_epoch = n_epochs

        self.optimizer = make_optimizer(optimizer)

        self.lr = learning_rate
        self.wd = weight_dacay

        self.early_stop = early_stop
        self.when = when_early_stop

        self.loss_func = make_loss_func(loss_func)()

        self.best_model_val_loss = None

        self.train_losses = []
        self.val_losses = []

        self.verbose = verbose

        self.save_log = save_train_log

        if save_train_log:
            self.train_log = ''



    def set_model(self, model: Module, model_name: str):
        self.model = model
        self.model_name = model_name + f"_seed_{self.seed}"
        self.optimizer = self.optimizer(self.model.parameters(),
                                        self.lr,
                                        weight_decay=self.wd)
        


    def train(self, 
              train, 
              val,
              param_path_dir: str=None
              ):


        assert isinstance(self.model, Module), 'The model is not uploaded to trainer'


        best_epoch = 0
        best_val_loss = 1e9

        self.path = self.model_name

        if param_path_dir:
            self.path = Path(param_path_dir) / self.path



        for i_epoch in range(self.n_epoch):
        
                
            train_losses = 0
            self.model.train()
               
            for batch in train:
                            
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
        
                y_true = batch.y
                y_hat = self.model(batch)
                loss = self.loss_func(y_hat.squeeze(-1), y_true)
                loss.backward()
                                    
                train_losses += loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()
        
            mean_train_loss = train_losses / (len(train))
        
        
            self.model.eval()
            mean_val_loss = self.test(val) / len(val)


            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss 
                best_epoch = i_epoch
                self.best_model_val_loss_param = self.model.state_dict()
                torch.save(self.best_model_val_loss_param, self.path)

                  
        
            if self.early_stop:
                if i_epoch - best_epoch > self.when:
                    break
        
            self.train_losses.append(mean_train_loss)
            self.val_losses.append(mean_val_loss)

            msg = f"Epoch {i_epoch+1:<4} | Train Loss {mean_train_loss:<6.5f} | Test Loss: {mean_val_loss:<6.5f} | Best Test Loss: {best_val_loss:<6.5f}"

            if self.verbose:
                print(msg)

            if self.save_log:
                self.train_log += msg + '\n'

        print()

    def test(self, val):

        val_losses = 0

        with torch.no_grad():
            for batch in val:

                batch = batch.to(self.device)        
                y_true = batch.y
                y_hat = self.model(batch)
                val_losses += self.loss_func(y_hat.squeeze(-1), y_true).item()

        return  val_losses




class TrainerData():
    def __init__(self,
                 train_data: str,
                 test_data: str,
                 save_parameter_dir: str,
                 seed: int,
                 batch_size: int=32,
                 test_frac: float=0.8):


        self.train_data = torch.load(train_data)
        self.test_data = torch.load(test_data)

        self.save_param_dir = save_parameter_dir

        self.seed = seed
        self.batch_size = batch_size
        self.test_frac = test_frac


    def prepare_splits(self):

        val_idx = split_data(self.train_data,
                             self.test_frac,
                             seed=self.seed)


        train_data = [data for i, data in enumerate(self.train_data) if i not in val_idx]
        val_data = [data for i, data in enumerate(self.train_data) if i in val_idx]

        return train_data, val_data


    def prepare_loaders(self):

        train_data, val_data = self.prepare_splits()

        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)





class Predictor():
    def __init__(self):

        self.metric_func = {'Rp': rp_calc,
                            'Rs': rs_calc,
                            'R2': r2_score,
                            'RMSE': rmse,
                            'MAE': mae}


    @staticmethod
    def predict(model: Module, test: DataLoader, model_params=None) -> Tuple[np.ndarray, np.ndarray]:

        device = next(model.parameters()).device

        if model_params is not None:
            model.load_state_dict(model_params)

        model.eval()

        with torch.no_grad():

            y_test = []
            y_hat = []
                               
            for batch in test:
                batch = batch.to(device)
                          
                y_true = batch.y
                y = model(batch)
        
                y_hat.append(y.detach().cpu().numpy())
                y_test.append(y_true.detach().cpu().numpy())

        y_hat = np.concatenate([a.flatten() for a in y_hat])
        y_test = np.concatenate([a.flatten() for a in y_test])
        
        return y_test, y_hat

    def calc_perf_stats(self, y_test, y_hat):

        stats_line = 'Test Prediction Stats: '
        metrics = ''

        for k, v in self.metric_func.items():

            value = v(y_test, y_hat)
            metrics += f"{k} = {value:.3f} | "


        stats_line += metrics + '\n'
        print(stats_line)

        
