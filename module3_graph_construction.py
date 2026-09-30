# =============================================================================
# MODULE 3: DTW-Based Graph Construction
# Builds inter-stock correlation graph using Dynamic Time Warping
# Input : processed_data.pkl  (from Module 2)
# Output: graph_data.pkl      (adjacency matrix + graph info)
# =============================================================================

import numpy as np
import pickle
import matplotlib.pyplot as plt
from dtaidistance import dtw

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
DTW_THRESHOLD = 0.7       # only keep edges where correlation >= 0.7
SIGMA         = 1.0       # Gaussian kernel width

# =============================================================================
# STEP 1: LOAD PROCESSED DATA
# =============================================================================

def load_processed_data(filepath='processed_data.pkl'):
    print("[Step 1] Loading processed data...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"  Stocks loaded : {len(data['stocks'])}")
    print(f"  Stocks        : {data['stocks']}")
    return data

# =============================================================================
# STEP 2: EXTRACT PRICE SERIES
# We use normalized price (column 0 of feature matrix) for DTW
# =============================================================================

def extract_price_series(data):
    print("\n[Step 2] Extracting normalized price series...")
    price_series = {}
    for stock in data['stocks']:
        # feature matrix shape: (T, 2) → column 0 is normalized price
        price_series[stock] = data['features'][stock][:, 0]
        print(f"  {stock}: {len(price_series[stock])} time steps")
    return price_series

# =============================================================================
# STEP 3: COMPUTE DTW DISTANCE MATRIX
# For every pair of stocks (i, j) — compute DTW distance
# 28 stocks → 28x28 = 784 pairs (378 unique pairs)
# =============================================================================

def compute_dtw_matrix(price_series, stocks):
    print("\n[Step 3] Computing DTW distance matrix...")
    print("  This may take a few minutes for 28 stocks...")

    n = len(stocks)
    dtw_matrix = np.zeros((n, n))

    total_pairs = (n * (n - 1)) // 2
    computed    = 0

    for i in range(n):
        for j in range(i + 1, n):
            s1 = price_series[stocks[i]].astype(np.double)
            s2 = price_series[stocks[j]].astype(np.double)

            # Compute DTW distance
            distance = dtw.distance_fast(s1, s2)

            dtw_matrix[i][j] = distance
            dtw_matrix[j][i] = distance   # symmetric

            computed += 1
            if computed % 50 == 0:
                print(f"  Progress: {computed}/{total_pairs} pairs computed")

    # Set diagonal to 0 (distance from stock to itself)
    np.fill_diagonal(dtw_matrix, 0)

    print(f"  ✓ DTW matrix computed — shape: {dtw_matrix.shape}")
    print(f"  Min distance: {dtw_matrix[dtw_matrix > 0].min():.4f}")
    print(f"  Max distance: {dtw_matrix.max():.4f}")
    return dtw_matrix

# =============================================================================
# STEP 4: CONVERT DTW DISTANCE → CORRELATION
# Smaller distance = more similar = higher correlation
# Formula: correlation = 1 / (1 + distance)
# =============================================================================

def distance_to_correlation(dtw_matrix):
    print("\n[Step 4] Converting DTW distances to correlations...")
    
    # Normalize distances to [0, 1] range first
    max_dist = dtw_matrix[dtw_matrix > 0].max()
    min_dist = dtw_matrix[dtw_matrix > 0].min()
    
    n = dtw_matrix.shape[0]
    corr_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i][j] = 1.0
            else:
                # Normalize then invert — smaller distance = higher correlation
                norm_dist = (dtw_matrix[i][j] - min_dist) / (max_dist - min_dist)
                corr_matrix[i][j] = 1 - norm_dist
    
    print(f"  Correlation range: [{corr_matrix[corr_matrix < 1].min():.4f}, "
          f"{corr_matrix[corr_matrix < 1].max():.4f}]")
    return corr_matrix

# =============================================================================
# STEP 5: APPLY THRESHOLD + GAUSSIAN KERNEL
# Only keep edges where correlation >= 0.7
# Weight edges using Gaussian kernel
# =============================================================================

def build_adjacency_matrix(corr_matrix, threshold=DTW_THRESHOLD, sigma=SIGMA):
    print(f"\n[Step 5] Building adjacency matrix (threshold={threshold})...")
    n = corr_matrix.shape[0]
    adj_matrix = np.zeros((n, n))

    edges = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if corr_matrix[i][j] >= threshold:
                # Apply Gaussian kernel to weight the edge
                d = 1 - corr_matrix[i][j]   # convert back to distance-like
                weight = np.exp(-(d ** 2) / (sigma ** 2))
                adj_matrix[i][j] = weight
                edges += 1

    print(f"  Total edges formed  : {edges}")
    print(f"  Avg edges per stock : {edges / n:.1f}")
    print(f"  Graph density       : {edges / (n * (n-1)):.2%}")
    return adj_matrix

# =============================================================================
# STEP 6: EXTRACT EDGE INDICES
# For GCN — we need list of (source, target) pairs
# =============================================================================

def extract_edge_indices(adj_matrix):
    print("\n[Step 6] Extracting edge indices...")
    edge_sources = []
    edge_targets = []

    n = adj_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] > 0:
                edge_sources.append(i)
                edge_targets.append(j)

    edge_indices = np.array([edge_sources, edge_targets])
    print(f"  Edge indices shape: {edge_indices.shape}")
    return edge_indices

# =============================================================================
# STEP 7: VISUALIZATIONS
# =============================================================================

def plot_dtw_heatmap(dtw_matrix, stocks, save=True):
    """Plot DTW distance heatmap between all stocks."""
    plt.figure(figsize=(14, 12))
    plt.imshow(dtw_matrix, cmap='hot_r', aspect='auto')
    plt.colorbar(label='DTW Distance')
    plt.xticks(range(len(stocks)), stocks, rotation=90, fontsize=8)
    plt.yticks(range(len(stocks)), stocks, fontsize=8)
    plt.title('DTW Distance Matrix — 28 NSE Energy Stocks')
    plt.tight_layout()
    if save:
        plt.savefig('dtw_distance_heatmap.png', dpi=150)
        print("  Saved: dtw_distance_heatmap.png")
    plt.show()

def plot_correlation_heatmap(corr_matrix, stocks, save=True):
    """Plot correlation heatmap between all stocks."""
    plt.figure(figsize=(14, 12))
    plt.imshow(corr_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(label='Correlation')
    plt.xticks(range(len(stocks)), stocks, rotation=90, fontsize=8)
    plt.yticks(range(len(stocks)), stocks, fontsize=8)
    plt.title('DTW Correlation Matrix — 28 NSE Energy Stocks')
    plt.tight_layout()
    if save:
        plt.savefig('dtw_correlation_heatmap.png', dpi=150)
        print("  Saved: dtw_correlation_heatmap.png")
    plt.show()

def plot_adjacency_heatmap(adj_matrix, stocks, save=True):
    """Plot final adjacency matrix heatmap."""
    plt.figure(figsize=(14, 12))
    plt.imshow(adj_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(label='Edge Weight')
    plt.xticks(range(len(stocks)), stocks, rotation=90, fontsize=8)
    plt.yticks(range(len(stocks)), stocks, fontsize=8)
    plt.title(f'Adjacency Matrix (threshold={DTW_THRESHOLD}) — 28 NSE Energy Stocks')
    plt.tight_layout()
    if save:
        plt.savefig('adjacency_matrix_heatmap.png', dpi=150)
        print("  Saved: adjacency_matrix_heatmap.png")
    plt.show()

# =============================================================================
# MAIN — RUN MODULE 3
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  MODULE 3: DTW-Based Graph Construction")
    print("=" * 60)

    # Step 1: Load processed data
    data   = load_processed_data('processed_data.pkl')
    stocks = data['stocks']

    # Step 2: Extract price series
    price_series = extract_price_series(data)

    # Step 3: Compute DTW distance matrix
    dtw_matrix = compute_dtw_matrix(price_series, stocks)

    # Step 4: Convert to correlation
    corr_matrix = distance_to_correlation(dtw_matrix)

    # Step 5: Build adjacency matrix
    adj_matrix = build_adjacency_matrix(corr_matrix)

    # Step 6: Extract edge indices
    edge_indices = extract_edge_indices(adj_matrix)

    # Step 7: Visualizations
    print("\n[Step 7] Generating plots...")
    plot_dtw_heatmap(dtw_matrix, stocks)
    plot_correlation_heatmap(corr_matrix, stocks)
    plot_adjacency_heatmap(adj_matrix, stocks)

    # Step 8: Save graph data
    print("\n[Step 8] Saving graph data...")
    graph_data = {
        'adj_matrix'  : adj_matrix,
        'corr_matrix' : corr_matrix,
        'dtw_matrix'  : dtw_matrix,
        'edge_indices': edge_indices,
        'stocks'      : stocks,
        'n_stocks'    : len(stocks),
        'threshold'   : DTW_THRESHOLD
    }
    with open('graph_data.pkl', 'wb') as f:
        pickle.dump(graph_data, f)
    print("  Saved: graph_data.pkl")
    print("  (This file will be loaded by Module 4 — GCN Feature Extraction)")

    print("\n" + "=" * 60)
    print("  MODULE 3 COMPLETE")
    print(f"  Stocks in graph : {len(stocks)}")
    print(f"  DTW threshold   : {DTW_THRESHOLD}")
    print(f"  Edges formed    : {(adj_matrix > 0).sum()}")
    print("=" * 60)