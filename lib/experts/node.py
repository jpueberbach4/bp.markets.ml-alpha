import torch
import torch.nn as nn
import torch.nn.functional as F
from tutel import moe as tutel_moe
from lib.experts.config import SquadConfig
from lib.experts.squads.registry import SquadRegistry
import logging
from dataclasses import dataclass


try:
    from lib.experts.routers.dense import DenseRouter
    from lib.experts.routers.dilated import DilatedMacroRouter
    import lib.experts.squads.null
    import lib.experts.squads.candlestick
    import lib.experts.squads.rsi
    import lib.experts.squads.indicator
except ImportError:
    # Optional modules may not be present depending on environment
    pass

logger = logging.getLogger("FeatureRoutedExpert")


@dataclass
class NetworkHyperparameters:
    """
    Container for network-level hyperparameters.

    Attributes:
        residual_dropout (float): Dropout for residual translators.
        local_proj_dropout (float): Dropout for local projection layers.
        moe_gate_noise (float): Noise applied to MoE gates.
        router_temp_train (float): Router softmax temperature during training.
        router_temp_eval (float): Router softmax temperature during evaluation.
    """
    residual_dropout: float = 0.1
    local_proj_dropout: float = 0.1
    moe_gate_noise: float = 0.1
    router_temp_train: float = 1.0
    router_temp_eval: float = 1.0


class ResidualTranslator(nn.Module):
    """
    Projects features between dimensions with optional residual connection.

    If input and output dimensions match, a residual connection is applied.
    """

    def __init__(self, in_d, out_d, dropout_rate):
        """
        Initializes the ResidualTranslator.

        Args:
            in_d (int): Input feature dimension.
            out_d (int): Output feature dimension.
            dropout_rate (float): Dropout probability.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_d),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_d, out_d),
            nn.GELU()
        )

        # Enable residual connection only when dimensions match
        self.is_residual = (in_d == out_d)

    def forward(self, x):
        """
        Forward pass through translator.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Transformed tensor (with residual if applicable).
        """
        return x + self.net(x) if self.is_residual else self.net(x)


class HMoENode(nn.Module):
    """
    Hierarchical Mixture-of-Experts (HMoE) node.

    Each node can either be a leaf (expert) or an internal router that
    aggregates outputs from child nodes using learned routing.
    """

    def __init__(self, node_path: str, config: SquadConfig):
        """
        Initializes an HMoENode.

        Args:
            node_path (str): Dot-separated path identifying this node.
            config (SquadConfig): Configuration for this node.
        """
        super().__init__()

        # Hyperparameters container
        self.hp = NetworkHyperparameters()

        # Node identity and configuration
        self.node_path = node_path
        self.config = config

        # Structural properties
        self.is_leaf = not bool(config.children)
        self.has_local = len(config.features) > 0
        self.fallback_squad = getattr(config, 'fallback_squad', None)

        # Classification metadata
        self.class_map = getattr(config, 'class_map', None)
        self.num_classes = getattr(config, 'num_classes', 3)

        # Expert configuration
        num_exp = getattr(config, 'num_experts', 4)
        k_val = min(2, num_exp)  # Top-k gating

        # Local feature projection
        layers = [
            nn.Linear(config.input_dim if config.input_dim > 0 else 1, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
            nn.Dropout(p=self.hp.local_proj_dropout)
        ]
        self.local_proj = nn.Sequential(*layers)

        # --- LEAF NODE SETUP ---
        if self.is_leaf:
            squad_name = node_path.split('.')[-1]

            # Try to resolve a registered squad implementation
            registry_key = next((key for key in SquadRegistry._squads.keys() if key in squad_name), None)

            if registry_key:
                self.physical_squad = SquadRegistry._squads[registry_key](config)
            else:
                # Default to generic Tutel MoE layer
                self.physical_squad = tutel_moe.moe_layer(
                    gate_type={'type': 'top', 'k': k_val, 'gate_noise': self.hp.moe_gate_noise},
                    model_dim=config.model_dim,
                    experts={
                        'type': 'ffn',
                        'hidden_size_per_expert': config.model_dim * 2,
                        'activation_fn': nn.GELU(),
                        'count_per_node': num_exp
                    }
                )

            # Final classifier for multi-class output
            self.classifier = nn.Linear(config.model_dim, self.num_classes)
            nn.init.constant_(self.classifier.bias, 0.0)
            return

        # --- INTERNAL NODE SETUP ---

        # Local context MoE
        self.local_moe = tutel_moe.moe_layer(
            gate_type={'type': 'top', 'k': k_val, 'gate_noise': self.hp.moe_gate_noise},
            model_dim=config.model_dim,
            experts={
                'type': 'ffn',
                'hidden_size_per_expert': config.model_dim * 2,
                'activation_fn': nn.GELU(),
                'count_per_node': num_exp
            }
        )

        # Child nodes and projection adapters
        self.children_nodes = nn.ModuleDict()
        self.children_projectors = nn.ModuleDict()

        for child_name, child_cfg in config.children.items():
            c_path = f"{node_path}.{child_name}" if node_path else child_name

            # Propagate classification metadata to children
            child_cfg.class_map = self.class_map
            child_cfg.num_classes = self.num_classes

            self.children_nodes[child_name] = HMoENode(c_path, child_cfg)
            self.children_projectors[child_name] = ResidualTranslator(
                child_cfg.model_dim,
                config.model_dim,
                dropout_rate=self.hp.residual_dropout
            )

        self.num_routing_children = len(config.children)

        # Router components (only if children exist)
        if self.num_routing_children > 0:
            self.router_moe = tutel_moe.moe_layer(
                gate_type={'type': 'top', 'k': k_val, 'gate_noise': self.hp.moe_gate_noise},
                model_dim=config.model_dim,
                experts={
                    'type': 'ffn',
                    'hidden_size_per_expert': config.model_dim * 2,
                    'activation_fn': nn.GELU(),
                    'count_per_node': num_exp
                }
            )

            # Dynamic router selection
            routing_mode = getattr(config, 'routing_mode', 'dense')
            dilation_rate = getattr(config, 'dilation_rate', 4)

            if routing_mode == "dilated":
                self.router_network = DilatedMacroRouter(
                    input_dim=config.input_dim,
                    seq_len=config.seq_len,
                    num_children=self.num_routing_children,
                    dilation_rate=dilation_rate
                )
            else:
                self.router_network = DenseRouter(
                    input_dim=config.input_dim,
                    seq_len=config.seq_len,
                    num_children=self.num_routing_children
                )

    def forward(self, features_dict, targets=None, pos_weight=None):
        """
        Forward pass through the HMoE node.

        Args:
            features_dict (dict): Mapping of node paths to feature tensors.
            targets (Tensor, optional): Supervision targets.
            pos_weight (Tensor, optional): Optional weighting (unused here).

        Returns:
            Tuple containing:
                - logits
                - features
                - auxiliary loss
                - task loss
                - entropy
                - routing statistics
        """
        aux_loss, entropy = 0.0, 0.0
        tokens_feat, tokens_logit, stats = [], [], {}

        device = next(self.parameters()).device

        # Retrieve or synthesize input sequence [B, SeqLen, Features]
        if self.config.input_dim == 0 or self.node_path not in features_dict:
            safe_dim = max(1, self.config.input_dim)
            B = next(iter(features_dict.values())).size(0) if features_dict else 1
            x_raw = torch.ones(B, self.config.seq_len, safe_dim, device=device)
        else:
            x_raw = features_dict[self.node_path]
            x_raw = torch.clamp(x_raw, min=-5.0, max=5.0)

        # Helper: extract last timestep or flatten sequence
        def flatten_seq(tensor):
            if tensor.dim() == 3:
                if tensor.size(1) == self.config.seq_len:
                    return tensor[:, -1, :]
                else:
                    return tensor[:, :, -1]
            return tensor

        # --- LEAF EXECUTION ---
        if self.is_leaf:
            if hasattr(self.physical_squad, 'input_projection'):
                # Sequence-based experts (e.g., CNNs)
                seq_x = x_raw
                if seq_x.size(1) == self.config.seq_len:
                    seq_x = seq_x.transpose(1, 2)
                local_feat = self.physical_squad(seq_x)
            else:
                # Dense experts (use latest timestep only)
                x_flat = flatten_seq(x_raw)
                if self.config.input_dim > 0:
                    x_flat = self.local_proj(x_flat)
                local_feat = self.physical_squad(x_flat)

            local_logit = self.classifier(local_feat)

            # Extract auxiliary loss if available
            aux_loss += getattr(self.physical_squad, 'l_aux', 0.0) if hasattr(self.physical_squad, 'l_aux') else getattr(getattr(self.physical_squad, 'moe_layer', None), 'l_aux', 0.0)

            return local_logit, local_feat, aux_loss, 0.0, entropy, stats

        # --- ROUTER EXECUTION ---

        # Local context representation
        x_flat = flatten_seq(x_raw)
        local_feat_context = self.local_moe(self.local_proj(x_flat))
        aux_loss += getattr(self.local_moe, 'l_aux', 0.0)

        child_names = list(self.children_nodes.keys())

        # Execute child nodes
        for child_name in child_names:
            c_logit, c_feat, c_aux, _, c_ent, c_stats = self.children_nodes[child_name](features_dict, targets, pos_weight)
            aux_loss += c_aux
            entropy += c_ent
            stats.update(c_stats)
            tokens_feat.append(self.children_projectors[child_name](c_feat))
            tokens_logit.append(c_logit)

        # Stack child outputs
        combined_feat = torch.stack(tokens_feat, dim=1)
        combined_logit = torch.stack(tokens_logit, dim=1)
        B, N, D = combined_feat.shape

        # Apply routing MoE to features
        routed_feat = self.router_moe(combined_feat.detach().view(B * N, D)).view(B, N, D)
        aux_loss += getattr(self.router_moe, 'l_aux', 0.0)

        # Prepare sequence input for router network
        x_seq = x_raw
        if x_seq.dim() == 2:
            x_seq = x_seq.unsqueeze(1).expand(-1, self.config.seq_len, -1)
        elif x_seq.dim() == 3 and x_seq.size(-1) == self.config.seq_len:
            x_seq = x_seq.transpose(1, 2)

        # Compute routing attention logits
        attn_logits = self.router_network(x_seq)

        # Apply temperature-scaled softmax
        current_temp = self.hp.router_temp_train if self.training else self.hp.router_temp_eval
        attn_weights = F.softmax(attn_logits / current_temp, dim=1)

        # --- ROUTER SUPERVISION ---
        task_loss = 0.0
        if self.training and targets is not None and self.class_map is not None:
            target_router_idx = torch.zeros(B, dtype=torch.long, device=device)
            valid_mask = torch.zeros(B, dtype=torch.bool, device=device)

            for raw_val, map_info in self.class_map.items():
                squad_name = map_info['squad']
                if squad_name in child_names and raw_val != 0.0:
                    mask = (targets.view(-1) == float(raw_val))
                    target_router_idx[mask] = child_names.index(squad_name)
                    valid_mask |= mask

            if valid_mask.any():
                task_loss = F.cross_entropy(attn_logits[valid_mask], target_router_idx[valid_mask])

        # Aggregate outputs using attention weights
        pooled_logit = torch.sum(combined_logit * attn_weights.unsqueeze(-1), dim=1)
        pooled_feat = torch.sum(routed_feat * attn_weights.unsqueeze(-1), dim=1)

        # Entropy regularization
        attn_w_flat = attn_weights.squeeze(-1) + 1e-8
        entropy += -torch.sum(attn_w_flat * torch.log2(attn_w_flat), dim=1).mean()

        # Collect routing statistics
        avg_attn = attn_weights.mean(dim=0).detach().cpu().numpy()
        prefix = self.node_path if self.node_path else "ROOT"

        for idx, c_name in enumerate(child_names):
            stats[f"attn_{prefix}->{c_name}"] = float(avg_attn[idx])

        return pooled_logit, pooled_feat, aux_loss, task_loss, entropy, stats