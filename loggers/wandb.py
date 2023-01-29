#! Since PyTorch Lightning loggers only work with Trainer API, we will define our own logging process
# Load .env to OS
from dotenv import load_dotenv
load_dotenv()
import wandb

class WandbLogger(object):
  def __init__(self, project: str, run_name: str, reinit: bool = False) -> None:
    self.project = project
    self.run_name = run_name
    self.reinit = reinit
    wandb.init(project = self.project, name = self.run_name, reinit = self.reinit)
    
  def log_metrics(self, metric_dict: dict, epoch: int):
    wandb.log({**metric_dict, 'epoch': epoch})
  
  def log_hyperparameters(self, config: dict):
    wandb.config.update(config)
  
  def alert(self): # This sends notification via Slack or Email in case of an error
    raise NotImplemented
    