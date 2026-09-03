import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree


def compute_spatial_features(df, config):
    window = config.get("neighborhood_window", 5)
    zone_rows = config.get("zone_rows", 4)
    zone_cols = config.get("zone_cols", 4)

    spatial_cols = [
        "old_fail_density_5x5", "old_fail_density_3x3", "old_fail_count_5x5",
        "valid_neighbor_count_5x5", "nearest_old_fail_dist",
        "radial_position", "angular_position", "norm_row", "norm_col",
        "is_edge_die", "edge_neighbor_count",
        "zone_old_fail_rate", "wafer_old_yield",
    ]

    results = np.zeros((len(df), len(spatial_cols)), dtype=np.float32)
    wafer_ids = df["wafer_id"].values
    unique_wafers = np.unique(wafer_ids)

    for wid in unique_wafers:
        mask = wafer_ids == wid
        idx = np.where(mask)[0]
        wdf = df.iloc[idx]
        rows = wdf["die_row"].values
        cols = wdf["die_col"].values
        old_labels = wdf["old_label"].values.astype(np.float32)

        max_r, max_c = rows.max() + 1, cols.max() + 1
        valid_grid = np.zeros((max_r, max_c), dtype=np.float32)
        old_fail_grid = np.zeros((max_r, max_c), dtype=np.float32)
        valid_grid[rows, cols] = 1.0
        old_fail_grid[rows, cols] = old_labels

        center_r = (max_r - 1) / 2.0
        center_c = (max_c - 1) / 2.0

        # Vectorized density computations
        fd5 = _local_density(old_fail_grid, valid_grid, window)
        fd3 = _local_density(old_fail_grid, valid_grid, 3)
        fc5 = _local_sum(old_fail_grid, window)
        vc5 = _local_sum(valid_grid, window)

        # Nearest old fail distance using KDTree
        old_fail_pos = np.column_stack(np.where(old_fail_grid > 0.5))
        die_pos = np.column_stack([rows, cols])
        if len(old_fail_pos) > 0:
            tree = cKDTree(old_fail_pos)
            nearest_dist, _ = tree.query(die_pos, k=1)
        else:
            max_rd = np.sqrt(center_r**2 + center_c**2)
            nearest_dist = np.full(len(rows), max_rd, dtype=np.float32)

        # Vectorized position features
        radial = np.sqrt((rows - center_r)**2 + (cols - center_c)**2)
        max_rd = np.sqrt(center_r**2 + center_c**2)
        if max_rd > 0:
            radial /= max_rd
        angular = np.arctan2(rows - center_r, cols - center_c)
        norm_r = (rows - center_r) / center_r if center_r > 0 else np.zeros_like(rows, dtype=np.float32)
        norm_c = (cols - center_c) / center_c if center_c > 0 else np.zeros_like(cols, dtype=np.float32)

        # Edge detection: count non-valid neighbors in 3x3
        non_valid = 1.0 - valid_grid
        # Pad to handle boundary
        padded = np.pad(non_valid, 1, mode='constant', constant_values=1.0)
        edge_count_grid = np.zeros((max_r, max_c), dtype=np.float32)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                edge_count_grid += padded[1+dr:max_r+1+dr, 1+dc:max_c+1+dc]
        edge_counts = edge_count_grid[rows, cols]
        is_edge = (edge_counts > 0).astype(np.float32)

        # Zone fail rate
        zone_grid = _zone_fail_rate(old_fail_grid, valid_grid, zone_rows, zone_cols, max_r, max_c)

        # Wafer old yield
        n_total = len(old_labels)
        wafer_yield = 1.0 - old_labels.sum() / n_total if n_total > 0 else 1.0

        # Assemble results for this wafer
        results[idx, 0] = fd5[rows, cols]
        results[idx, 1] = fd3[rows, cols]
        results[idx, 2] = fc5[rows, cols]
        results[idx, 3] = vc5[rows, cols]
        results[idx, 4] = nearest_dist.astype(np.float32)
        results[idx, 5] = radial.astype(np.float32)
        results[idx, 6] = angular.astype(np.float32)
        results[idx, 7] = norm_r.astype(np.float32)
        results[idx, 8] = norm_c.astype(np.float32)
        results[idx, 9] = is_edge
        results[idx, 10] = edge_counts
        results[idx, 11] = zone_grid[rows, cols]
        results[idx, 12] = wafer_yield

    return pd.DataFrame(results, columns=spatial_cols, index=df.index)


def _local_density(fail_grid, valid_grid, window):
    fail_sum = uniform_filter(fail_grid.astype(np.float64), size=window, mode='constant', cval=0.0)
    valid_sum = uniform_filter(valid_grid.astype(np.float64), size=window, mode='constant', cval=0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(valid_sum > 0, fail_sum / valid_sum, 0.0)
    return density.astype(np.float32)


def _local_sum(grid, window):
    s = uniform_filter(grid.astype(np.float64), size=window, mode='constant', cval=0.0)
    return (s * window * window).astype(np.float32)


def _zone_fail_rate(fail_grid, valid_grid, zone_rows, zone_cols, max_r, max_c):
    result = np.zeros((max_r, max_c), dtype=np.float32)
    row_edges = np.linspace(0, max_r, zone_rows + 1, dtype=int)
    col_edges = np.linspace(0, max_c, zone_cols + 1, dtype=int)
    for zr in range(zone_rows):
        for zc in range(zone_cols):
            rs, re = row_edges[zr], row_edges[zr+1]
            cs, ce = col_edges[zc], col_edges[zc+1]
            v = valid_grid[rs:re, cs:ce].sum()
            f = fail_grid[rs:re, cs:ce].sum()
            rate = f / v if v > 0 else 0.0
            result[rs:re, cs:ce] = rate
    return result
