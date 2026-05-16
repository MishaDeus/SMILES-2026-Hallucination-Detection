"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch


import torch

def aggregate(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

    layer1 = hidden_states[13]
    layer2 = hidden_states[14]
    layer3 = hidden_states[15]

    layer_data = (layer1 + layer2 + layer3) / 3.0

    real_indices = attention_mask.nonzero()[:, 0]
    window_size = 19
    response_indices = real_indices[-window_size:] if len(real_indices) >= window_size else real_indices
    
    response_tokens = layer_data[response_indices]
    
    mean_feat = response_tokens.mean(dim=0)
    if response_tokens.size(0) > 1:
        std_feat = response_tokens.std(dim=0)
    else:
        std_feat = torch.zeros_like(mean_feat)

    return torch.cat([mean_feat, std_feat])
    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:

    device = hidden_states.device
    real_indices = attention_mask.nonzero()[:, 0]
    window_size = 19

    response_indices = real_indices[-window_size:] if len(real_indices) >= window_size else real_indices

    response_states = hidden_states[:, response_indices, :]

    layer_norms = torch.norm(response_states, p=2, dim=-1).mean(dim=-1)

    drift_features = []
    n_layers = response_states.size(0)
    
    for i in range(n_layers - 1):
        layer_i_mean = response_states[i].mean(dim=0)
        layer_next_mean = response_states[i+1].mean(dim=0)

        sim = torch.nn.functional.cosine_similarity(layer_i_mean.unsqueeze(0), layer_next_mean.unsqueeze(0))
        drift_features.append(sim)
        
    drift_tensor = torch.cat(drift_features)

    seq_len = torch.tensor([float(len(real_indices))], device=device)

    return torch.cat([layer_norms, drift_tensor, seq_len])
    # ------------------------------------------------------------------

def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
