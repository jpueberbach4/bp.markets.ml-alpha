import os
import argparse
import sys
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# ==========================================
# DIAGNOSTIC LOGGING CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)-15s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True 
)

from lib.experts.routed import FeatureRoutedExpert
from lib.experts.data.pipeline import MoEDataPipeline

import config

logger = logging.getLogger("Visualizer")

CHECKPOINT_PATH = "./checkpoints/hierarchical_moe/Feature_Routed_MoE_ep1.pt"

SYMBOL = config.SYMBOL
TIMEFRAME = config.TIMEFRAME
START_DATE_VAL = config.START_DATE_VAL
END_DATE_VAL = config.END_DATE_VAL

TARGET_FEATURE = config.TARGET_FEATURE

def _set_input_dims_recursively(squads_dict, feature_map, prefix=""):
    """
    Recursively updates the topology configuration to match the exact dimensions of the fetched live data.

    This ensures that the loaded model weights map perfectly to the incoming tensors, 
    preventing dimension mismatch errors if certain API features failed to fetch during ingestion.

    Args:
        squads_dict (dict): The nested dictionary of SquadConfig objects.
        feature_map (dict): A dictionary mapping node paths to successfully fetched feature lists.
        prefix (str, optional): The accumulated dot-notation path of the current node. Defaults to "".
    """
    for k, v in squads_dict.items():
        node_path = f"{prefix}.{k}" if prefix else k
        if node_path in feature_map:
            v.input_dim = len(feature_map[node_path])
        if v.children:
            _set_input_dims_recursively(v.children, feature_map, node_path)

def main():
    """
    The primary execution entry point for the HMoE Inference Visualizer.

    This script performs several critical diagnostic functions:
    1. Loads a pre-trained Hierarchical Mixture of Experts checkpoint.
    2. Runs a forward pass on validation data, using hooks to intercept the macro-router's delegation probabilities.
    3. Performs an aggressive "Occlusion Attribution" pass, systematically zeroing out input features to measure their impact on the final prediction.
    4. Calculates a raw, event-aware F1 score (without Non-Maximum Suppression).
    5. Renders a comprehensive, interactive 3-panel Plotly dashboard to visualize the network's internal decision-making process.
    """
    parser = argparse.ArgumentParser(description="MoE Raw Inference Visualizer")
    parser.add_argument("--checkpoint", type=str, help="Path to the .pt checkpoint file.")
    parser.add_argument("--threshold", type=float, help="Optional override for the decision threshold.")
    args = parser.parse_args()

    target_checkpoint = args.checkpoint if args.checkpoint else CHECKPOINT_PATH
    DEVICE = config.DEVICE

    if not os.path.exists(target_checkpoint):
        logger.error(f"Checkpoint not found at: {target_checkpoint}")
        return

    logger.info("Fetching Validation data for raw inference...")

    pipeline = MoEDataPipeline()
    
    # Fetch validation data. include_ohlcv=True is critical here to render the background candlestick chart.
    squad_tensors, Y_val, _, aligned_ohlcv = pipeline.fetch_squads_and_window(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=START_DATE_VAL,
        end_date=END_DATE_VAL,
        squads_config=config.SQUADS,  
        target_feature=TARGET_FEATURE,
        include_ohlcv=True
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
    
    checkpoint = torch.load(target_checkpoint, map_location=DEVICE, weights_only=False)
    
    saved_feature_names = checkpoint.get('feature_names', None)
    if not saved_feature_names:
        saved_feature_names = pipeline.last_squad_map
            
    _set_input_dims_recursively(expert_config.squads, pipeline.last_squad_map)

    # Clean state dict keys in case the model was saved while wrapped in torch.compile()
    raw_state_dict = checkpoint['model_state_dict']
    clean_state_dict = {}
    for k, v in raw_state_dict.items():
        clean_key = k.replace('_orig_mod.', '')
        clean_state_dict[clean_key] = v

    expert = FeatureRoutedExpert(config=expert_config).to(DEVICE)
    expert.load_state_dict(clean_state_dict, strict=False)
    expert.eval()

    threshold = args.threshold if args.threshold is not None else checkpoint.get('best_threshold', 0.5)

    all_preds = []
    all_attributions = []
    all_routings = []

    logger.info("Running chunked parallel inference and extracting Router Gates...")
    
    with torch.no_grad():
        for batch in val_loader:
            for k, v in batch['features'].items():
                batch['features'][k] = v.to(DEVICE)
            if 'labels' in batch and 'reversal_signal' in batch['labels']:
                batch['labels']['reversal_signal'] = batch['labels']['reversal_signal'].to(DEVICE)

            def routing_hook(module, input, output):
                """
                Surgically intercepts the raw logits from the Root Router, applies softmax to convert 
                them into probabilities, and saves them to memory to map the network's delegation strategy.
                """
                probs = F.softmax(output, dim=1).cpu().numpy()
                all_routings.append(probs)

            # Register the hook, run the baseline forward pass, and immediately remove the hook to prevent memory leaks.
            handle = expert.root_node.router_network.register_forward_hook(routing_hook)
            baseline_preds, _ = expert.forward_and_loss(batch, compute_loss=False)
            handle.remove() 

            batch_size = baseline_preds.size(0)
            batch_attrs = {feature: np.zeros(batch_size) for squad_list in saved_feature_names.values() for feature in squad_list}

            # =========================================================================
            # FAST OCCLUSION ATTRIBUTION
            # =========================================================================
            # Instead of complex gradient-based attribution (like Integrated Gradients), this uses 
            # absolute occlusion. It zeroes out chunks of features, passes them through the network, 
            # and measures how drastically the output probabilities change. High deviation = High importance.
            MAX_FEATURES_PER_CHUNK = 16

            for node_path, feature_list in saved_feature_names.items():
                if node_path not in batch['features']: 
                    continue
                
                pristine_tensor = batch['features'][node_path]
                num_features = pristine_tensor.size(1)

                for chunk_start in range(0, num_features, MAX_FEATURES_PER_CHUNK):
                    chunk_end = min(chunk_start + MAX_FEATURES_PER_CHUNK, num_features)
                    chunk_size = chunk_end - chunk_start

                    # Expand the batch to process multiple occlusions in a single parallel GPU pass
                    expanded_tensor = pristine_tensor.unsqueeze(0).expand(chunk_size, -1, -1, -1).clone()
                    
                    for i, f_idx in enumerate(range(chunk_start, chunk_end)):
                        expanded_tensor[i, :, f_idx, :] = 0.0

                    mega_batch_tensor = expanded_tensor.view(chunk_size * batch_size, num_features, -1)

                    mega_batch = {'features': {}}
                    for k, v in batch['features'].items():
                        if k == node_path:
                            mega_batch['features'][k] = mega_batch_tensor
                        else:
                            mega_batch['features'][k] = v.repeat(chunk_size, 1, 1)

                    occluded_preds_mega, _ = expert.forward_and_loss(mega_batch, compute_loss=False)
                    occluded_preds = occluded_preds_mega.view(chunk_size, batch_size, -1)

                    # Calculate the deviation impact against the baseline prediction
                    for i, f_idx in enumerate(range(chunk_start, chunk_end)):
                        feature_name = feature_list[f_idx] if f_idx < len(feature_list) else f"{node_path}_{f_idx}"
                        impact = torch.abs(baseline_preds - occluded_preds[i]).sum(dim=-1)
                        batch_attrs[feature_name] = impact.cpu().numpy()

            all_preds.append(baseline_preds.cpu())
            
            for i in range(batch_size):
                step_attr = {k: batch_attrs[k][i] for k in batch_attrs.keys() if k in batch_attrs}
                all_attributions.append(step_attr)

    predictions = torch.cat(all_preds).numpy() 
    routing_probs = np.vstack(all_routings)    
    
    labels = Y_val.squeeze().numpy()
    
    # Extract prediction channels based on the multi-task MoE configuration
    prob_chop = predictions[:, 0]
    prob_top = predictions[:, 1]
    prob_bot = predictions[:, 2]

    class_targets = np.zeros_like(labels, dtype=int)
    class_targets[labels == 1.0] = 1   
    class_targets[labels == -1.0] = 2  

    # =========================================================================
    # RAW EVENT-AWARE SCORING (NO NMS)
    # =========================================================================
    # This evaluates the network's raw firing accuracy across continuous event zones,
    # treating a cluster of adjacent target labels as a single cohesive event.
    tp, fp, fn, total_events = 0.0, 0.0, 0.0, 0.0

    for idx, (raw_val, name) in zip([1, 2], [(1.0, 'Top'), (-1.0, 'Bottom')]):
        c_preds = predictions[:, idx]
        c_strict_targets = (labels == raw_val).astype(float)
        
        raw_fires_c = (c_preds >= threshold).astype(float)
        
        shifted_c = np.concatenate(([0.0], c_strict_targets[:-1]))
        starts_c = (c_strict_targets == 1.0) & (shifted_c == 0.0)
        event_ids_c = np.cumsum(starts_c) * c_strict_targets
        c_events = int(np.sum(starts_c))
        total_events += c_events
        
        hit_mask = (raw_fires_c == 1.0) & (c_strict_targets == 1.0)
        hit_ids = np.unique(event_ids_c[hit_mask])
        hit_ids = hit_ids[hit_ids > 0]
        
        c_tp = float(len(hit_ids))
        c_fn = float(c_events - c_tp)
        c_fp = float(np.sum(raw_fires_c * (1.0 - c_strict_targets)))
        
        tp += c_tp
        fp += c_fp
        fn += c_fn

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_score = 2.0 * (precision * recall) / (precision + recall + 1e-8)

    logger.info(f"--- Multi-Class Event Detection Summary (NO NMS) ---")
    logger.info(f"Target Zones Found: {int(total_events)}")
    logger.info(f"Performance -> Event TPs: {int(tp)} | Raw FPs: {int(fp)} | Precision: {precision*100:.1f}% | Recall: {recall*100:.1f}%")
    logger.info(f"Final Aggregate Event F1 Score: {f1_score:.4f}")
    logger.info(f"---------------------------------------------")

    # Construct the Plotly Hover Data to expose the internal state vectors on the chart
    hover_texts = []
    for i in range(len(predictions)):
        attr = all_attributions[i]
        total_impact = sum(attr.values()) + 1e-9
        sorted_attr = sorted(attr.items(), key=lambda x: x[1], reverse=True)
        
        txt = f"<b>Prediction Profile:</b><br>"
        txt += f" - Top Prob: {prob_top[i]:.3f}<br>"
        txt += f" - Bot Prob: {prob_bot[i]:.3f}<br>"
        txt += f" - Chop Prob: {prob_chop[i]:.3f}<br><br>"
        txt += f"<b>Router Delegation:</b><br>"
        txt += f" - Top Squad: {routing_probs[i, 0]:.1%}<br>" 
        txt += f" - Bottom Squad: {routing_probs[i, 1]:.1%}<br>" 
        txt += f" - Null Squad: {routing_probs[i, 2]:.1%}<br><br>" 
        txt += f"<b>Top Features (Cross-Squad):</b><br>"
        
        for k, v in sorted_attr[:15]:
            pct = (v / total_impact) * 100
            if pct >= 0.0: 
                txt += f"{k}: <b>{pct:.1f}%</b><br>"
        hover_texts.append(txt)

    # =========================================================================
    # PLOTLY DASHBOARD CONSTRUCTION
    # =========================================================================
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.25, 0.15])
    x_indices = np.arange(len(aligned_ohlcv["close"]))
    offset_dist = np.mean(aligned_ohlcv["close"]) * 0.0015

    # Panel 1: Price Action and Trigger Overlays
    fig.add_trace(go.Candlestick(x=x_indices, open=aligned_ohlcv["open"], high=aligned_ohlcv["high"], low=aligned_ohlcv["low"], close=aligned_ohlcv["close"], name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    
    actual_bull_idx = np.where(labels == -1.0)[0]
    fig.add_trace(go.Scatter(x=actual_bull_idx, y=aligned_ohlcv["low"][actual_bull_idx] - offset_dist, mode='markers', marker=dict(symbol='star', color='yellow', size=12, line=dict(width=1, color='black')), name='Actual Bottoms (Bull)'), row=1, col=1)

    actual_bear_idx = np.where(labels == 1.0)[0]
    fig.add_trace(go.Scatter(x=actual_bear_idx, y=aligned_ohlcv["high"][actual_bear_idx] + offset_dist, mode='markers', marker=dict(symbol='star', color='orange', size=12, line=dict(width=1, color='black')), name='Actual Tops (Bear)'), row=1, col=1)

    fire_top_idx = np.where(prob_top >= threshold)[0]
    fire_bot_idx = np.where(prob_bot >= threshold)[0]

    tp_top_idx = fire_top_idx[labels[fire_top_idx] == 1.0]
    fp_top_idx = fire_top_idx[labels[fire_top_idx] != 1.0]
    
    tp_bot_idx = fire_bot_idx[labels[fire_bot_idx] == -1.0]
    fp_bot_idx = fire_bot_idx[labels[fire_bot_idx] != -1.0]

    fig.add_trace(go.Scatter(x=tp_bot_idx, y=aligned_ohlcv["low"][tp_bot_idx] - (offset_dist * 2), mode='markers', marker=dict(symbol='triangle-up', color='cyan', size=14, line=dict(width=1, color='black')), name='Model BUY (TP Bull)', text=[hover_texts[i] for i in tp_bot_idx], hovertemplate="%{text}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=tp_top_idx, y=aligned_ohlcv["high"][tp_top_idx] + (offset_dist * 2), mode='markers', marker=dict(symbol='triangle-down', color='red', size=14, line=dict(width=1, color='black')), name='Model SELL (TP Bear)', text=[hover_texts[i] for i in tp_top_idx], hovertemplate="%{text}<extra></extra>"), row=1, col=1)

    fig.add_trace(go.Scatter(x=fp_bot_idx, y=aligned_ohlcv["close"][fp_bot_idx], mode='markers', marker=dict(symbol='triangle-up-open', color='cyan', size=14, line=dict(width=2, color='cyan')), name='Model BUY (FP Bull)', text=[hover_texts[i] for i in fp_bot_idx], hovertemplate="%{text}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=fp_top_idx, y=aligned_ohlcv["close"][fp_top_idx], mode='markers', marker=dict(symbol='triangle-down-open', color='red', size=14, line=dict(width=2, color='red')), name='Model SELL (FP Bear)', text=[hover_texts[i] for i in fp_top_idx], hovertemplate="%{text}<extra></extra>"), row=1, col=1)

    # Panel 2: Continuous Probability Tracking
    fig.add_trace(go.Scatter(x=x_indices, y=prob_top, mode='lines', name='Prob: Top', line=dict(color='red', width=1.5), text=hover_texts, hovertemplate="%{text}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_indices, y=prob_bot, mode='lines', name='Prob: Bot', line=dict(color='cyan', width=1.5), text=hover_texts, hovertemplate="%{text}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_indices, y=prob_chop, mode='lines', name='Prob: Chop', line=dict(color='gray', width=1.0, dash='dot'), text=hover_texts, hovertemplate="%{text}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="white", row=2, col=1, opacity=0.5)

    # Panel 3: Discrete Target Profiles
    fig.add_trace(go.Scatter(x=x_indices, y=labels, mode='lines', name='Target Profile (1=Top, -1=Bottom, 0=Chop)', line=dict(color='dodgerblue', width=2, shape='hv')), row=3, col=1)
    fig.add_hline(y=1.0, line_dash="dot", line_color="green", row=3, col=1, opacity=0.5)
    fig.add_hline(y=0.0, line_dash="solid", line_color="gray", row=3, col=1, opacity=0.3)
    fig.add_hline(y=-1.0, line_dash="dot", line_color="red", row=3, col=1, opacity=0.5)

    title_str = f"Multi-Class HMoE Map | Aggregate Event F1: {f1_score:.4f} (Prec: {precision*100:.1f}%, Rec: {recall*100:.1f}%) | Threshold: {threshold:.3f}"
    fig.update_layout(title=title_str, template="plotly_dark", hovermode="closest", height=1500, xaxis_rangeslider_visible=False)
    fig.show()

if __name__ == "__main__":
    main()