import os
import argparse
import torch
import torch.nn as nn
import logging
import sys

import lib.experts.logging.tutelspam
import lib.experts.logging.registry

from lib.experts.routed import FeatureRoutedExpert, HMoENode
from lib.experts.data.pipeline import MoEDataPipeline

import config

logger = logging.getLogger("Verify")

SYMBOL = config.SYMBOL
TIMEFRAME = config.TIMEFRAME
START_DATE_VAL = config.START_DATE_VAL
END_DATE_VAL = config.END_DATE_VAL
TARGET_FEATURE = config.TARGET_FEATURE # Updated to match config

routing_stats = {}
attention_stats = {}

def get_routing_hook(node_path):
    def hook(module, input, output):
        winning_experts = torch.argmax(output, dim=-1).view(-1)
        unique, counts = torch.unique(winning_experts, return_counts=True)

        if node_path not in routing_stats:
            routing_stats[node_path] = {}

        for u, c in zip(unique.tolist(), counts.tolist()):
            routing_stats[node_path][u] = routing_stats[node_path].get(u, 0) + c

    return hook

def _set_input_dims_recursively(squads_dict, feature_map, prefix=""):
    for k, v in squads_dict.items():
        node_path = f"{prefix}.{k}" if prefix else k
        if node_path in feature_map:
            v.input_dim = len(feature_map[node_path])
        if v.children:
            _set_input_dims_recursively(v.children, feature_map, node_path)

def _log_tree(node: HMoENode, indent: str = ""):
    node_name = node.node_path if node.node_path else "ROOT_ROUTER"
    feat_info = f" (+ {len(node.config.features)} local features)" if node.has_local else ""
    router_info = " [ROUTER]" if not node.is_leaf else " [LEAF]"
    
    logger.info(f"{indent}➔ {node_name}{router_info}{feat_info}")
    if not node.is_leaf:
        for child_node in node.children_nodes.values():
            _log_tree(child_node, indent + "    ")

def main():
    parser = argparse.ArgumentParser(description="HMoE Live Data Load Balancer Verifier")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file.")
    args = parser.parse_args()

    DEVICE = config.DEVICE

    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found at: {args.checkpoint}")
        return

    logger.info(f"Fetching Live Data ({START_DATE_VAL} to {END_DATE_VAL}) to test Router behavior...")

    pipeline = MoEDataPipeline()

    squad_tensors, Y_val, _ = pipeline.fetch_squads_and_window(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=START_DATE_VAL,
        end_date=END_DATE_VAL,
        squads_config=config.SQUADS,
        target_feature=TARGET_FEATURE
    )

    if not squad_tensors:
        logger.error("No data fetched. Check your connection or dates.")
        return

    val_loader = pipeline.build_dataloader(
        squad_tensors=squad_tensors,
        Y=Y_val,
        batch_size=64,
        is_training=False
    )

    expert_config = config.get_expert_config()
    _set_input_dims_recursively(expert_config.squads, pipeline.last_squad_map)

    expert = FeatureRoutedExpert(config=expert_config).to(DEVICE)
    
    logger.info("=== Loaded Hierarchical MoE Tree ===")
    _log_tree(expert.root_node)
    logger.info("====================================")

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    
    raw_state_dict = checkpoint.get('model_state_dict', checkpoint)
    clean_state_dict = {}
    for k, v in raw_state_dict.items():
        clean_key = k.replace('_orig_mod.', '')
        clean_state_dict[clean_key] = v

    expert.load_state_dict(clean_state_dict, strict=False)
    expert.eval()

    logger.info("Attaching surgical diagnostic hooks to HMoE Nodes...")

    hooks_attached = 0
    for name, module in expert.named_modules():
        if 'gate' in name.lower() and isinstance(module, nn.Linear):
            
            node_path = "UNKNOWN"
            if 'root_node.router_moe' in name:
                node_path = "ROOT_ROUTER"
            elif 'root_node.local_moe' in name:
                node_path = "ROOT_LOCAL_FEATURES"
            elif 'children_nodes.' in name:
                parts = name.split('children_nodes.')
                last_part = parts[-1]
                sub_path = last_part.split('.')[0]
                
                # Reconstruct full path
                full_path_parts = []
                for p in parts[1:]:
                    full_path_parts.append(p.split('.')[0])
                node_path = ".".join(full_path_parts)
                
                if 'router_moe' in name:
                    node_path += "_ROUTER"
                elif 'local_moe' in name:
                    node_path += "_LEAF"

            module.register_forward_hook(get_routing_hook(node_path))
            hooks_attached += 1
            logger.info(f"Hook successfully attached to layer: {name} -> mapping to [{node_path}]")

    if hooks_attached == 0:
        logger.error("No hooks attached! Check topology.")
        return

    logger.info(f"Attached {hooks_attached} hooks. Flowing tokens through the HMoE...")

    batch_count = 0
    with torch.no_grad():
        for batch in val_loader:
            for k, v in batch['features'].items():
                batch['features'][k] = v.to(DEVICE)
            
            # Safely map targets to DEVICE
            if 'labels' in batch and 'reversal_signal' in batch['labels']:
                batch['labels']['reversal_signal'] = batch['labels']['reversal_signal'].to(DEVICE)
            
            _, loss_dict = expert.forward_and_loss(batch, compute_loss=True)
            
            for k, v in loss_dict.items():
                if k.startswith("attn_"):
                    # Accumulate attention stats correctly
                    attention_stats[k] = attention_stats.get(k, 0.0) + v
            batch_count += 1

    print("\n" + "=" * 80)
    print(f"{'HMoE LIVE LOAD BALANCING & ATTENTION REPORT':^80}")
    print("=" * 80)

    print("\n--- EXPERT ROUTING DISTRIBUTION ---")
    for node in sorted(routing_stats.keys()):
        stats = routing_stats[node]
        total_windows = sum(stats.values())

        print(f"\nNODE: [{node.upper()}] | Total Tokens Routed: {total_windows}")
        print("-" * 60)

        for expert_id in sorted(stats.keys()):
            count = stats[expert_id]
            percentage = (count / total_windows) * 100
            bar = "█" * int(percentage / 2)
            print(f" Expert {expert_id:<2} | {percentage:05.2f}% | {bar}")

    print("\n" + "=" * 80)
    print("\n--- NODE ATTENTION WEIGHTS (MACRO ROUTING) ---")
    
    if batch_count > 0:
        for k, v in sorted(attention_stats.items()):
            avg_attn = (v / batch_count) * 100
            bar = "█" * int(avg_attn / 2)
            print(f" {k:<40} | {avg_attn:05.2f}% | {bar}")
    else:
        print("No attention stats collected.")

    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()