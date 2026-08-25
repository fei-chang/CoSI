import os
import time
from datetime import datetime
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.distributed as dist
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf

from .logger import setup_logger, log_message
from .evaluator import Evaluator

from utils import pretty_print_losses
from data import build_dataloader
from models import build_model

def train_ddp(rank, world_size, cfg):
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

    # Set device
    device = torch.device(f'cuda:{rank}')

    # Create Trainer instance
    trainer = Trainer(cfg, rank, device)
    trainer.train()

    # Clean up
    dist.destroy_process_group()

class Trainer:
    def __init__(self, cfg, rank=0, device=None):
        self.rank = rank
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.distributed = cfg.distributed
        self.device = torch.device(cfg.device) if device is None else device
        
        self.epochs = cfg.stage.num_epochs
        self.save_every = cfg.stage.save_every
    
        # Set up save directory
        if self.rank == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            self.save_dir = os.path.join(
                cfg.save_dir,
                cfg.model.model_name,
                cfg.stage.stage_name,
                f"run_{timestamp}"
            )

            os.makedirs(self.save_dir, exist_ok=True)
            self.save_best_only = cfg.stage.save_best_only
            self.save_best_metrics = cfg.stage.save_best_metrics
            
            self.logger = setup_logger(self.save_dir)

        self.train_loader = build_dataloader(cfg, cfg.stage.dataset_name, 
                                            is_train=True, 
                                            batch_size=cfg.stage.batch_size,
                                            num_workers=cfg.stage.num_workers,
                                            pin_memory=cfg.stage.pin_memory,
                                            prefetch_factor=cfg.stage.prefetch_factor,
                                            persistent_workers=cfg.stage.persistent_workers,
                                            distributed=cfg.distributed)
                                            
        self.val_loader =  build_dataloader(cfg, cfg.stage.dataset_name, 
                                            is_train=False, 
                                            batch_size=cfg.stage.batch_size,
                                            num_workers=cfg.stage.num_workers,
                                            pin_memory=cfg.stage.pin_memory,
                                            prefetch_factor=cfg.stage.prefetch_factor,
                                            persistent_workers=cfg.stage.persistent_workers,
                                            distributed=cfg.distributed)
        
        self.model = build_model(self.device, cfg, verbose=(cfg.verbose_load&(self.rank==0)))
        
        if self.world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device])
            print(f"Rank {self.rank}: Model wrapped with DistributedDataParallel")
        else:
            print(f"Rank {self.rank}: Model NOT wrapped with DistributedDataParallel")

        
        self.total_steps = len(self.train_loader) * self.epochs
        self.warmup_steps = len(self.train_loader) * cfg.stage.optimizer.warm_up_epochs  

        model_to_configure = self.model.module if hasattr(self.model, "module") else self.model

        if cfg.stage.freeze_gaze_backbone:
            model_to_configure.freeze_gaze_backbone()

        model_to_configure.freeze_dino_backbone()
        if cfg.model.freeze_spatial:
            model_to_configure.freeze_spatial()
                
        trainable_params = []
        trainable_param_names = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_param_names.append(name)
                trainable_params.append(param)
        self.trainable_param_names = trainable_param_names

        self.optimizer = optim.Adam(
            trainable_params,
            lr=cfg.stage.optimizer.lr,
            weight_decay=cfg.stage.optimizer.weight_decay
        )

            
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )

        self.loss_dict = []
        self.best_score = float('-inf')        
        self.evaluater = Evaluator(self.model, 
                                   self.device,
                                   cfg.stage.pattern_type)

        self.evaluater.load_data(cfg.stage.dataset_name, 
                                 self.val_loader,
                                 cfg.stage.weights.pattern>0,
                                 cfg.stage.weights.inout>0
                                )
        
        self.evaluater.enable_logging = False

        if self.rank == 0:
            self.log_configuration(cfg)
        

    def log_configuration(self, cfg):
        config_msg = self._build_config_message(cfg)
        log_message(self.logger, config_msg)

    def _build_config_message(self, cfg):
        # Basic Train Configuration
        msg = (
            f"\n{'='*50}\n"
            f"Training Configuration:\n"
            f"Model: {cfg.model.model_name}\n"            
            f"Train Mode: {cfg.stage.stage_name}\n"
            f"Dataset: {cfg.stage.dataset_name}\n"
            f"Batch Size: {cfg.stage.batch_size}\n"
            f"Learning Rate: {cfg.stage.optimizer.lr}\n"
            f"Epochs: {cfg.stage.num_epochs}\n"
            f"Save Best Metrics: {cfg.stage.save_best_metrics}\n"
            f"Save Every: {cfg.stage.save_every} epochs\n"
            f"Warmup Epochs: {cfg.stage.optimizer.warm_up_epochs}\n"
            f"Pretrained Weights From: {cfg.pretrained_weights}\n"
        )
    
        # Detailed Model Configuration
        msg += "\nModel Configuration:\n"
        msg += f"{OmegaConf.to_yaml(cfg.model)}\n"  

        # Check Updated Components
        msg += "\nTrainable Parameters:\n"
        if hasattr(self, 'trainable_param_names'):
            for name in self.trainable_param_names:
                msg += f"- {name}\n"
        
        msg += f"{'='*50}\n"
        return msg

    def _store_weights(self):
        self._stored_weights = [
            param.detach().clone() for param in self.model.parameters()
            if param.requires_grad
        ]

    def _validate_weights(self):
        current_weights = [
            param for param in self.model.parameters() 
            if param.requires_grad
        ]
        for stored, current in zip(self._stored_weights, current_weights):
            if not torch.allclose(stored, current, rtol=1e-08, atol=1e-08):
                print("Weights updated.")
                return
        print("Problem, weights not updated.")

    def train(self, check_initial:bool=False):
        self.start_time = time.time()
        if (self.rank == 0)&(check_initial):
            self.validate_and_save('untrained')

        for epoch in range(self.epochs):
            # self._store_weights()
            self.train_epoch(epoch)
            if (epoch + 1) % self.save_every == 0:
                if self.rank == 0:
                    self.validate_and_save(epoch, save_best_only=self.save_best_only)
            # self._validate_weights()

        if self.rank == 0:
            return self.best_score

    def train_epoch(self, epoch):
        self.model.train()
        print_every = 100
        counter = 0
        for batch in tqdm(self.train_loader, 
                          desc=f"[Epoch:{epoch+1}/{self.epochs}]",
                          disable=(self.rank != 0)):
            counter +=1
            self.optimizer.zero_grad()
            self.loss_dict = self.model(batch)
            self.loss_dict['total'].backward()
            self.optimizer.step()
            self.scheduler.step()
            if counter%print_every == 0:
                print(self.loss_dict)
                
        if self.rank == 0:
            self.log_epoch(epoch)
    
    def log_epoch(self, epoch):
        elapsed_time = time.time() - self.start_time
        remaining_epochs = self.epochs - (epoch + 1)
        eta = (elapsed_time / (epoch + 1)) * remaining_epochs
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        loss_msg = pretty_print_losses(self.loss_dict)
        msg = f'Epoch [{epoch + 1}/{self.epochs}], ETA: {eta_str}\n{loss_msg}\n'
        log_message(self.logger, msg)

    def validate_and_save(self, epoch, save_best_only=True):

        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        tmp_model_path = os.path.join(self.save_dir, 'tmp_model_%s.pth'%epoch)

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model_to_save.state_dict(), tmp_model_path)

        metrics = self.evaluater.evaluate(
            model_id=epoch,
            metric_save_path=None
        )

        self.logger.log_metrics(metrics, epoch)
        test_score = metrics[self.save_best_metrics]

        if test_score > self.best_score:
            self.best_score = test_score
            torch.save(model_to_save.state_dict(), best_model_path)
            msg = f'New best model saved with test score: {test_score:.4f}\n'
            log_message(self.logger, msg)

        if save_best_only:
            os.remove(tmp_model_path)



        
