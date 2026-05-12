from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    
    n_samples = len(y)
    indices = np.arange(n_samples)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    results = []
    
    for train_val_idx, test_idx in skf.split(indices, y):
        y_train_val = y[train_val_idx]

        from sklearn.model_selection import train_test_split
        actual_val_size = max(0.1, val_size) 
        
        idx_train, idx_val = train_test_split(
            train_val_idx,
            test_size=actual_val_size,
            random_state=random_state,
            stratify=y_train_val
        )
        
        results.append((idx_train, idx_val, test_idx))
        
    return results