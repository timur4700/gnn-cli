from dataclasses import dataclass, asdict, field
from utils.func import save_json


@dataclass
class Config:
    optimizer: str=field(default='adam',
                          metadata={
                              'choices': ['adam', 'adam_w'],
                              'editable': True
                          })

    seed: int=field(default=1,
                          metadata={
                              'min': 1,
                              'max': 999999,
                              'editable': True  
                          })

    

    n_epochs: int=field(default=30,
                          metadata={
                              'min': 1,
                              'max': 999999,
                              'editable': True
                          })


    learning_rate: float=field(default=1e-3,
                          metadata={
                              'min': 1e-6,
                              'max': 1e-1,
                              'editable': True
                          })

    weight_dacay: float=field(default=0,
                          metadata={
                              'min': 0,
                              'max': 1e-1,
                              'editable': True
                          })

    early_stop: bool=field(default=False,
                          metadata={
                              'choices': [True, False],
                              'editable': True
                          })  


      
    when_early_stop: float=field(default=100,
                          metadata={
                              'min': 1,
                              'max': 2**32,
                              'editable': True
                          })


    
    loss_func: str=field(default='mse',
                          metadata={
                              'choices': ['mse'],
                              'editable': True
                          })


    device: str=field(default='cpu',
                          metadata={
                              'choices': ['cpu', 'cuda'],
                              'editable': True
                          })


    verbose: bool=field(default=1,
                          metadata={
                              'choices': [0, 1],
                              'editable': True
                          })   


    save_train_log: bool=field(default=False,
                          metadata={
                              'choices': [True, False],
                              'editable': True
                          })


@dataclass
class TrainerData:
    train_data: str=field(default='',
                          metadata={
                              'editable': False
                          })
    
    test_data: str=field(default='',
                          metadata={
                              'editable': False
                          })


    save_parameter_dir: str=field(default='',
                          metadata={
                              'editable': False
                          })



    seed: int=field(default=1,
                          metadata={
                              'min': 1,
                              'max': 999999,
                              'editable': True  
                          })
    

    batch_size: int=field(default=32,
                          metadata={
                              'min': 1,
                              'max': 4096,
                              'editable': True
                          })

    test_frac: float=field(default=0.1,
                          metadata={
                              'min': 0,
                              'max': 1,
                              'editable': True
                          })


@dataclass
class TrainerConfig:
    config: Config=field(default_factory=Config)
    data: TrainerData=field(default_factory=TrainerData)



    def set_data(self, train: str, test: str, param_dir: str):

        self.data.train_data = train
        self.data.test_data = test
        self.data.save_parameter_dir = param_dir

    def save(self, wd):
        save_json(asdict(self), wd)




    

    
