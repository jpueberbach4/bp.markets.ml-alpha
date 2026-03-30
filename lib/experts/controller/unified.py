import torch
import os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from lib.experts.base import Expert
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger("Controller")

class UnifiedTrainingController:
    """
    Manages the complete training, validation, and evaluation lifecycle for a Hierarchical Mixture of Experts (HMoE) model.

    This controller handles device mapping, dynamic Non-Maximum Suppression (NMS) for event isolation,
    multi-class tolerant F1 scoring, learning rate scheduling, and metadata-rich checkpointing.
    """

    def __init__(
        self,
        expert: Expert,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        checkpoint_dir: str = './checkpoints',
        feature_names: dict = None
    ):
        """
        Initializes the UnifiedTrainingController.

        Args:
            expert (Expert): The PyTorch neural network model (HMoE) to be trained.
            train_loader (DataLoader): The data loader for the training dataset.
            val_loader (DataLoader): The data loader for the validation dataset.
            device (torch.device): The hardware device (CPU/GPU) to run computations on.
            checkpoint_dir (str, optional): Directory path to save model checkpoints. Defaults to './checkpoints'.
            feature_names (dict, optional): Dictionary mapping node paths to their specific feature lists. Defaults to None.

        Raises:
            ValueError: If either the train_loader or val_loader is not provided.
        """
        if train_loader is None:
            raise ValueError("CRITICAL: 'train_loader' passed to UnifiedTrainingController is None. Check train.py!") 
        if val_loader is None:
            raise ValueError("CRITICAL: 'val_loader' passed to UnifiedTrainingController is None. Check train.py!") 

        self.expert = expert.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.feature_names = feature_names

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = self.expert.configure_optimizer()

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=8,
            verbose=True
        )

        self.best_val_threshold = 0.5
        logger.info(f"Initialized Unified Controller for Expert: [{self.expert.expert_id}] on {self.device}.")

    def _prepare_payload(self, raw_payload: dict) -> dict:
        """
        Recursively traverses the input payload and moves all PyTorch tensors to the target hardware device.

        Args:
            raw_payload (dict): The raw batch dictionary yielded by the DataLoader.

        Returns:
            dict: A new dictionary with all tensor values mapped to the specified device.
        """
        prepared = {}
        for key, value in raw_payload.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, dict):
                prepared[key] = self._prepare_payload(value)
            else:
                prepared[key] = value
        return prepared

    def _apply_dynamic_nms(self, probs: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Applies a 1D Non-Maximum Suppression (NMS) algorithm to isolate peak event probabilities.
        
        This prevents the model from triggering multiple consecutive signals for a single market event
        by reducing continuous blocks of high probability into a single localized spike.

        Args:
            probs (torch.Tensor): A 1D tensor of predicted probabilities for a specific class.
            threshold (float): The minimum probability required to consider a prediction active.

        Returns:
            torch.Tensor: A binary tensor of the same shape where only the local maxima within active islands are set to 1.0.
        """
        nms_probs = probs.clone()
        active = (nms_probs >= threshold).float()

        if active.max() == 0.0:
            return torch.zeros_like(nms_probs)

        shifted = F.pad(active[:-1], (1, 0), mode='constant', value=0.0)
        starts = (active == 1.0) & (shifted == 0.0)

        island_ids = torch.cumsum(starts.float(), dim=0) * active

        result = torch.zeros_like(nms_probs)
        num_islands = int(island_ids.max().item())

        for i in range(1, num_islands + 1):
            island_mask = (island_ids == i)
            island_probs = nms_probs * island_mask
            max_idx = torch.argmax(island_probs)
            result[max_idx] = 1.0

        return result

    def train_epoch(self, epoch_idx: int):
        """
        Executes a single pass of the training loop over the entire training dataset.

        Evaluates a static threshold to monitor general epoch health, calculates the multi-class 
        Cross-Entropy loss, propagates gradients, and performs an intensive grid-search to find 
        the optimal evaluation threshold via event-based NMS.

        Args:
            epoch_idx (int): The current epoch number (used for logging and display).

        Returns:
            float: The average composite loss over the entire training epoch.
        """
        self.expert.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        global_tp = 0.0
        global_fp = 0.0
        global_fn = 0.0
        threshold = self.best_val_threshold

        loop = tqdm(self.train_loader, desc=f"Epoch {epoch_idx} (Train)", unit="batch")

        for batch_idx, raw_payload in enumerate(loop):
            payload = self._prepare_payload(raw_payload)
            self.optimizer.zero_grad()

            predictions, loss_dict = self.expert.forward_and_loss(payload)

            if not loss_dict:
                continue

            combined_loss = sum(v for k, v in loss_dict.items() if 'loss' in k)
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.expert.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += combined_loss.item()
            raw_targets = payload['labels']['reversal_signal'].view(-1)
            
            class_targets = torch.zeros_like(raw_targets, dtype=torch.long)
            for raw_val, map_info in self.expert.root_node.config.class_map.items():
                class_targets[raw_targets == float(raw_val)] = map_info['idx']

            with torch.no_grad():
                all_preds.append(predictions.detach().cpu())
                all_targets.append(raw_targets.cpu())

                batch_max_p = predictions[:, 1:].detach().max().item()
                batch_entropy = loss_dict.get('entropy', torch.tensor(0.0)).item()
                
                for map_info in self.expert.root_node.config.class_map.values():
                    idx = map_info['idx']
                    if idx == 0: continue
                    
                    prob_c = predictions[:, idx].detach()
                    pred_bin_c = (prob_c > threshold).float()
                    target_bin_c = (class_targets == idx).float()
                    
                    global_tp += (pred_bin_c * target_bin_c).sum().item()
                    global_fp += (pred_bin_c * (1 - target_bin_c)).sum().item()
                    global_fn += ((1 - pred_bin_c) * target_bin_c).sum().item()

                precision = global_tp / (global_tp + global_fp + 1e-8)
                recall = global_tp / (global_tp + global_fn + 1e-8)
                running_f1 = 2.0 * (precision * recall) / (precision + recall + 1e-8)

            loop.set_postfix({
                'loss': f"{combined_loss.item():.4f}",
                'f1': f"{running_f1:.4f}",
                'ent': f"{batch_entropy:.4f}",
                'max_p': f"{batch_max_p:.2f}"
            })

        avg_loss = total_loss / len(self.train_loader)
        full_preds = torch.cat(all_preds, dim=0).squeeze()
        full_targets = torch.cat(all_targets, dim=0).squeeze()

        max_pred = max(0.51, full_preds[:, 1:].max().item())
        threshold_grid = torch.linspace(0.10, max_pred, steps=21)
        
        best_f1, best_precision, best_recall = 0.0, 0.0, 0.0

        for thresh in threshold_grid:
            tp, fp, fn = 0.0, 0.0, 0.0
            
            for raw_val, map_info in self.expert.root_node.config.class_map.items():
                idx = map_info['idx']
                if idx == 0: continue 
                
                c_preds = full_preds[:, idx]
                c_strict_targets = (full_targets == float(raw_val)).float()
                
                nms_spikes = self._apply_dynamic_nms(c_preds, thresh.item())
                
                shifted_c = torch.cat([torch.tensor([0.0]), c_strict_targets[:-1]])
                starts_c = ((c_strict_targets == 1.0) & (shifted_c == 0.0)).float()
                event_ids_c = torch.cumsum(starts_c, dim=0) * c_strict_targets
                total_events_c = int(starts_c.sum().item())
                
                shifted_l = torch.roll(c_strict_targets, shifts=-1, dims=0)
                shifted_r = torch.roll(c_strict_targets, shifts=1, dims=0)
                shifted_l[-1] = 0.0; shifted_r[0] = 0.0
                tolerant_targets_c = torch.clamp(c_strict_targets + shifted_l + shifted_r, 0.0, 1.0)
                
                ids_l = torch.roll(event_ids_c, shifts=-1, dims=0)
                ids_r = torch.roll(event_ids_c, shifts=1, dims=0)
                ids_l[-1] = 0.0; ids_r[0] = 0.0
                tolerant_ids_c = torch.max(event_ids_c, torch.max(ids_l, ids_r))
                
                hit_mask = (nms_spikes == 1.0) & (tolerant_targets_c == 1.0)
                hit_ids = torch.unique(tolerant_ids_c[hit_mask])
                hit_ids = hit_ids[hit_ids > 0]
                
                c_tp = float(hit_ids.numel())
                c_fn = float(total_events_c - c_tp)
                c_fp = (nms_spikes * (1.0 - tolerant_targets_c)).sum().item()
                
                tp += c_tp
                fn += c_fn
                fp += c_fp
                
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1_score = 2.0 * (precision * recall) / (precision + recall + 1e-8)
            
            if f1_score > best_f1:
                best_f1, best_precision, best_recall = f1_score, precision, recall

        logger.info(f"Epoch {epoch_idx} (Train) Completed | Avg Loss: {avg_loss:.6f} | Tolerant F1: {best_f1:.4f} (Prec: {best_precision*100:.1f}%, Rec: {best_recall*100:.1f}%)")
        return avg_loss

    def validate(self, epoch_idx: int):
        """
        Executes a single pass of the evaluation loop over the validation dataset.

        Gathers predictions without calculating gradients. Performs an intensive granular threshold
        search to determine the most statistically optimal threshold for the current network weights.

        Args:
            epoch_idx (int): The current epoch number.

        Returns:
            tuple: A tuple containing:
                - avg_loss (float): The average validation loss.
                - best_f1 (float): The optimal tolerant F1 score achieved across the threshold grid.
        """
        self.expert.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        loop = tqdm(self.val_loader, desc=f"Epoch {epoch_idx} (Valid)", unit="batch")

        with torch.no_grad():
            for batch_idx, raw_payload in enumerate(loop, start=1):
                payload = self._prepare_payload(raw_payload)
                predictions, loss_dict = self.expert.forward_and_loss(payload, compute_loss=True)
                
                targets = payload['labels']['reversal_signal'].view(-1)

                combined_loss = loss_dict.get('total_loss', torch.tensor(0.0, device=self.device))
                total_loss += combined_loss.item()

                all_preds.append(predictions.cpu())
                all_targets.append(targets.cpu())

                loop.set_postfix({
                    'val_loss': f"{(total_loss / batch_idx):.4f}",
                    'tuning': '...'
                })

        full_preds = torch.cat(all_preds, dim=0).squeeze()
        full_targets = torch.cat(all_targets, dim=0).squeeze()

        max_pred = max(0.51, full_preds[:, 1:].max().item())
        threshold_grid = torch.linspace(0.10, max_pred, steps=51)
        
        best_f1, best_thresh, best_precision, best_recall = 0.0, 0.01, 0.0, 0.0

        for thresh in threshold_grid:
            tp, fp, fn = 0.0, 0.0, 0.0
            
            for raw_val, map_info in self.expert.root_node.config.class_map.items():
                idx = map_info['idx']
                if idx == 0: continue 
                
                c_preds = full_preds[:, idx]
                c_strict_targets = (full_targets == float(raw_val)).float()
                
                nms_spikes = self._apply_dynamic_nms(c_preds, thresh.item())
                
                shifted_c = torch.cat([torch.tensor([0.0]), c_strict_targets[:-1]])
                starts_c = ((c_strict_targets == 1.0) & (shifted_c == 0.0)).float()
                event_ids_c = torch.cumsum(starts_c, dim=0) * c_strict_targets
                total_events_c = int(starts_c.sum().item())
                
                shifted_l = torch.roll(c_strict_targets, shifts=-1, dims=0)
                shifted_r = torch.roll(c_strict_targets, shifts=1, dims=0)
                shifted_l[-1] = 0.0; shifted_r[0] = 0.0
                tolerant_targets_c = torch.clamp(c_strict_targets + shifted_l + shifted_r, 0.0, 1.0)
                
                ids_l = torch.roll(event_ids_c, shifts=-1, dims=0)
                ids_r = torch.roll(event_ids_c, shifts=1, dims=0)
                ids_l[-1] = 0.0; ids_r[0] = 0.0
                tolerant_ids_c = torch.max(event_ids_c, torch.max(ids_l, ids_r))
                
                hit_mask = (nms_spikes == 1.0) & (tolerant_targets_c == 1.0)
                hit_ids = torch.unique(tolerant_ids_c[hit_mask])
                hit_ids = hit_ids[hit_ids > 0]
                
                c_tp = float(hit_ids.numel())
                c_fn = float(total_events_c - c_tp)
                c_fp = (nms_spikes * (1.0 - tolerant_targets_c)).sum().item()
                
                tp += c_tp
                fn += c_fn
                fp += c_fp
                
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1_score = 2.0 * (precision * recall) / (precision + recall + 1e-8)
            
            if f1_score > best_f1:
                best_f1, best_thresh, best_precision, best_recall = f1_score, thresh.item(), precision, recall

        self.best_val_threshold = best_thresh
        avg_loss = total_loss / len(self.val_loader)

        logger.info(
            f"Epoch {epoch_idx} (Valid) Completed | Avg Loss: {avg_loss:.6f} | "
            f"Tolerant F1: {best_f1:.4f} (Prec: {best_precision*100:.1f}%, Rec: {best_recall*100:.1f}%) (at threshold: {best_thresh:.3f})"
        )
        
        self.scheduler.step(best_f1)
        return avg_loss, best_f1

    def save_checkpoint(self, epoch_idx: int, loss: float, metric: float):
        """
        Saves the complete state of the model to disk, including metadata required for 
        visualizer tools and inference loading.

        Args:
            epoch_idx (int): The current epoch number, used to construct the filename.
            loss (float): The validation loss, used to construct the filename.
            metric (float): The F1 metric, used to construct the filename.
        """
        file_name = f"{self.expert.expert_id}_ep{epoch_idx}_loss{loss:.4f}_f1_{metric:.4f}.pt"
        path = os.path.join(self.checkpoint_dir, file_name)

        def _extract_meta(squads_dict):
            meta = {}
            for k, v in squads_dict.items():
                meta[k] = {'input_dim': v.input_dim, 'num_experts': v.num_experts, 'seq_len': v.seq_len}
                if v.children:
                    meta[k]['children'] = _extract_meta(v.children)
            return meta

        squad_metadata = {}
        target_config = getattr(self.expert, 'full_config', None)

        if target_config and hasattr(target_config, 'squads'):
            squad_metadata = _extract_meta(target_config.squads)

        checkpoint_bundle = {
            'epoch': epoch_idx,
            'expert_id': self.expert.expert_id,
            'model_state_dict': self.expert.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': loss,
            'val_f1': metric, 
            'best_threshold': self.best_val_threshold,
            'squad_metadata': squad_metadata,
            'feature_names': self.feature_names
        }

        torch.save(checkpoint_bundle, path)
        logger.info(f"Checkpoint saved with metadata: {path}")

    def run_full_training(self, max_epochs: int, patience: int = 7):
        """
        The primary orchestration method that loops through the desired number of epochs.
        
        Handles learning rate scheduling updates, noise gating updates inside the expert layers, 
        and triggers early stopping if the validation metric stalls.

        Args:
            max_epochs (int): The maximum absolute number of epochs to run.
            patience (int, optional): Number of epochs to wait for an F1 improvement before aborting. Defaults to 7.
        """
        best_val_metric = -1.0
        epochs_no_improve = 0

        for epoch in range(1, max_epochs + 1):
            if hasattr(self.expert, 'update_gate_noise'):
                self.expert.update_gate_noise(epoch, max_epochs)

            train_loss = self.train_epoch(epoch)
            val_loss, val_metric = self.validate(epoch)

            self.save_checkpoint(epoch, val_loss, val_metric)

            if val_metric > best_val_metric:
                best_val_metric = val_metric
                epochs_no_improve = 0
                logger.info(f"New best validation F1: {best_val_metric:.4f}. Resetting patience.")
            else:
                epochs_no_improve += 1
                logger.info(f"Validation F1 did not improve. Patience: {epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                logger.warning(f"Early stopping triggered after {epoch} epochs.")
                break