import os, torch, tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from metrics.criterion import GWDetectionCriterion
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from matplotlib import pyplot as plt
from typing import Any, Dict, Iterable, Tuple

class TrainingPipeline:
    def __init__(self, 
                model: nn.Module,
                lossfunc: GWDetectionCriterion,
                optimizer: torch.optim.Optimizer,
                *,
                device: str="cpu", 
                weight_init: bool=True,
                custom_weight_initializer: Any=None,
                dirname: str="./saved_model", 
                filename: str="model.pth.tar",
                save_metrics: bool=True):
        
        self.device = device
        self.model = model.to(self.device)
        self.lossfunc = lossfunc
        self.optimizer = optimizer
        self.weight_init = weight_init
        self.custom_weight_initializer = custom_weight_initializer
        self.dirname = dirname
        self.filename = filename
        self.save_metrics = save_metrics
        
        if self.weight_init:
            if self.custom_weight_initializer:
                self.model.apply(self.custom_weight_initializer)
            else:
                self.model.apply(self.xavier_init_weights)
        # collect metrics in this dictionary
        if self.save_metrics:
            self._train_metrics_dict = dict(bbox_loss=[], confidence_loss=[], total_loss=[])
            self._eval_metrics_dict = dict(bbox_loss=[], confidence_loss=[], total_loss=[])
        

    def xavier_init_weights(self, m: nn.Module):
        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)) and (m.weight.requires_grad == True):
            nn.init.xavier_uniform_(m.weight)
            if torch.is_tensor(m.bias):
                m.bias.data.fill_(0.01)
    

    def save_model(self):
        if not os.path.isdir(self.dirname): os.mkdir(self.dirname)
        state_dicts = {
            "network_params":self.model.state_dict(),
            "optimizer_params":self.optimizer.state_dict(),
        }
        return torch.save(state_dicts, os.path.join(self.dirname, self.filename))
    

    def collect_metric(self) -> Tuple[Dict[str, Iterable[float]], Dict[str, Iterable[float]]]:
        if self.save_metrics:
            return self._train_metrics_dict, self._eval_metrics_dict


    def plot_metrics(
            self, 
            mode: str,
            figsize: Tuple[float, float]=(20, 6)):
        
        valid_modes = self._valid_modes()
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")
        
        _, axs = plt.subplots(1, 3, figsize=figsize)
        axs[0].plot(getattr(self, f"_{mode}_metrics_dict")["bbox_loss"])
        axs[0].set_title(f"{mode} Bounding Box Loss")

        axs[1].plot(getattr(self, f"_{mode}_metrics_dict")["confidence_loss"])
        axs[1].set_title(f"{mode} Confidence Loss")

        axs[2].plot(getattr(self, f"_{mode}_metrics_dict")["total_loss"])
        axs[2].set_title(f"{mode} Total Loss")
        plt.show()
        print("\n\n")
        

    def train(self, dataloader: DataLoader, verbose: bool=False) -> Tuple[float, float, float]:
        bbox_loss, confidence_loss, total_loss = self._feed(dataloader, "train", verbose)
        return bbox_loss, confidence_loss, total_loss
    

    def evaluate(self, dataloader: DataLoader, verbose: bool=False) -> Tuple[float, float, float]:        
        with torch.no_grad():
            bbox_loss, confidence_loss, total_loss = self._feed(dataloader, "eval", verbose)
            return bbox_loss, confidence_loss, total_loss
        

    def _feed(self, dataloader: DataLoader, mode: str, verbose: bool=False) -> Tuple[float, float, float]:
        assert mode in self._valid_modes(), "Invalid Mode"
        getattr(self.model, mode)()
        bbox_loss, confidence_loss, total_loss = 0, 0, 0
        
        for idx, (signals, labels) in tqdm.tqdm(enumerate(dataloader)):
            signals = signals.to(self.device)       #shape: (N, n_channels, n_time)
            labels = labels.to(self.device)         #shape: (N, 1)    
                        
            preds = self.model(signals)
            batch_bbox_loss, batch_confidence_loss = self.lossfunc(preds, labels)
            batch_total_loss = batch_bbox_loss + batch_confidence_loss
            
            if mode == "train":
                batch_total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            bbox_loss += batch_bbox_loss.item()
            confidence_loss += batch_confidence_loss.item()
            total_loss += batch_total_loss

        bbox_loss /= (idx + 1)
        confidence_loss /= (idx + 1)
        total_loss /= (idx + 1)

        verbosity_label = mode.title()
        if verbose:
            print((
                f"{verbosity_label} BBox Loss: {round(bbox_loss, 4)}\
                  \t{verbosity_label} Confidence Loss: {round(bbox_loss, 4)}\
                  \t{verbosity_label} Total Loss: {round(total_loss, 4)}"
            ))
            
        if self.save_metrics:
            getattr(self, f"_{mode}_metrics_dict")["bbox_loss"].append(bbox_loss)
            getattr(self, f"_{mode}_metrics_dict")["confidence_loss"].append(confidence_loss)
            getattr(self, f"_{mode}_metrics_dict")["total_loss"].append(total_loss)

        return bbox_loss, confidence_loss, total_loss
    

    def _valid_modes(self) -> Iterable[str]:
        return ["train", "eval"]