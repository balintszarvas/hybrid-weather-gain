"""Estimate FLOPs for the hybrid model (dycore + GNN) on different grid types.

This provides theoretical estimates. For precise measurements, use JAX profiling.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

# Grid configurations for TL truncation
# For triangular truncation TL_N: approx lon_nodes = 3*N, lat_nodes = (3*N)/2 + 1
# Using equiangular_with_poles spacing

GRID_CONFIGS = {
    "TL31":  {"lon": 96,  "lat": 49,  "truncation": 31},
    "TL47":  {"lon": 144, "lat": 73,  "truncation": 47},
    "TL63":  {"lon": 192, "lat": 97,  "truncation": 63},
    "TL95":  {"lon": 288, "lat": 145, "truncation": 95},
    "TL127": {"lon": 384, "lat": 193, "truncation": 127},
}

# Model configuration (from experiment.py defaults)
DT_MODEL = 360  # seconds per dycore step
CORRECTION_INTERVAL = 3600  # seconds between GNN corrections
TOTAL_TIME_6H = 6 * 60 * 60  # 21600 seconds

# GNN configuration (from twod_gnn_correction.py)
GNN_HIDDEN_DIMS = (64, 64)  # default
NUM_VERTICAL_LEVELS = 8  # typical for your setup
NUM_PROGNOSTIC_VARS = 4  # vorticity, divergence, temperature, log_surface_pressure


@dataclass
class FLOPEstimate:
    """Container for FLOP estimates."""
    dycore_per_step: float
    dycore_total: float
    gnn_per_correction: float
    gnn_total: float
    total: float
    
    def __str__(self):
        return (
            f"  Dycore per step:     {self.dycore_per_step:.2e} FLOPs\n"
            f"  Dycore total (60x):  {self.dycore_total:.2e} FLOPs\n"
            f"  GNN per correction:  {self.gnn_per_correction:.2e} FLOPs\n"
            f"  GNN total (6x):      {self.gnn_total:.2e} FLOPs\n"
            f"  TOTAL (6h step):     {self.total:.2e} FLOPs"
        )


def estimate_spectral_transform_flops(truncation: int, lon: int, lat: int) -> float:
    """Estimate FLOPs for spherical harmonic transform (forward or inverse).
    
    The transform complexity is approximately O(T^2 * N) where T is truncation
    and N is number of grid points, dominated by Legendre transforms.
    
    More precisely:
    - FFT in longitude: O(lon * lat * log(lon)) 
    - Legendre transform: O(T^2 * lat) per longitude wavenumber
    """
    n_points = lon * lat
    
    # FFT: 5 * n * log2(n) FLOPs (standard estimate)
    fft_flops = 5 * lon * np.log2(lon) * lat
    
    # Legendre transform: O(T^2) per latitude ring, summed over wavenumbers
    # Roughly 2 * T^2 * lat * T = 2 * T^3 * lat (matrix-vector products)
    legendre_flops = 2 * (truncation ** 2) * lat * truncation
    
    return fft_flops + legendre_flops


def estimate_dycore_step_flops(lon: int, lat: int, truncation: int, 
                                num_levels: int = NUM_VERTICAL_LEVELS) -> float:
    """Estimate FLOPs for one dycore timestep.
    
    The dycore (primitive equations with semi-implicit time stepping) involves:
    1. Spectral transforms (forward and inverse) for each variable
    2. Non-linear term evaluation in physical space
    3. Semi-implicit linear solve in spectral space
    4. Hyperdiffusion filter
    """
    n_points = lon * lat
    n_spectral = (truncation + 1) * (truncation + 2) // 2  # triangular truncation
    
    # Number of prognostic fields (per level + surface)
    num_3d_fields = 3  # vorticity, divergence, temperature
    num_2d_fields = 1  # log_surface_pressure
    total_fields = num_3d_fields * num_levels + num_2d_fields
    
    # 1. Spectral transforms (forward + inverse per field, per RK stage)
    # IMEX-RK-SIL3 has 3 stages
    num_rk_stages = 3
    transform_flops = estimate_spectral_transform_flops(truncation, lon, lat)
    total_transform_flops = 2 * transform_flops * total_fields * num_rk_stages
    
    # 2. Non-linear terms in physical space
    # Includes: velocity computation, advection, Coriolis, etc.
    # Estimate: ~50 FLOPs per grid point per level per stage
    nonlinear_flops = 50 * n_points * num_levels * num_rk_stages
    
    # 3. Semi-implicit solve (tridiagonal system per spectral coefficient)
    # ~10 FLOPs per spectral coefficient per level
    implicit_flops = 10 * n_spectral * num_levels * num_rk_stages
    
    # 4. Hyperdiffusion filter (spectral multiplication)
    # ~2 FLOPs per spectral coefficient per field
    filter_flops = 2 * n_spectral * total_fields
    
    total = total_transform_flops + nonlinear_flops + implicit_flops + filter_flops
    
    return total


def estimate_gnn_layer_flops(num_nodes: int, num_edges: int, 
                              input_dim: int, output_dim: int) -> float:
    """Estimate FLOPs for one GNN layer.
    
    Each layer performs:
    1. Edge update: MLP on (edge, sender, receiver) → new edge
    2. Edge aggregation: sum edges per node
    3. Node update: MLP on (node, aggregated_edges) → new node
    
    For Dense layer: 2 * input * output FLOPs (multiply-add)
    """
    # Edge MLP: input = edge_dim + 2*node_dim, output = output_dim
    edge_input_dim = input_dim + 2 * input_dim  # edges + senders + receivers
    edge_mlp_flops = 2 * edge_input_dim * output_dim * num_edges
    
    # Edge aggregation (segment_sum): num_edges additions
    aggregation_flops = num_edges * output_dim
    
    # Node MLP: input = node_dim + edge_dim (after aggregation), output = output_dim
    node_input_dim = input_dim + output_dim
    node_mlp_flops = 2 * node_input_dim * output_dim * num_nodes
    
    # ReLU activations: 1 FLOP per element
    activation_flops = (num_edges + num_nodes) * output_dim
    
    return edge_mlp_flops + aggregation_flops + node_mlp_flops + activation_flops


def estimate_gnn_correction_flops(lon: int, lat: int, 
                                   hidden_dims: Tuple[int, ...] = GNN_HIDDEN_DIMS,
                                   num_levels: int = NUM_VERTICAL_LEVELS) -> float:
    """Estimate FLOPs for one GNN correction step.
    
    The GNN correction involves:
    1. Feature extraction from state
    2. Multiple GNN layers
    3. Output layer to produce corrections
    4. Conversion back to modal space (spectral transform)
    """
    num_nodes = lon * lat
    
    # 9-point stencil: approximately 9 edges per interior node
    # Boundary nodes have fewer (poles have ~7 on average)
    avg_edges_per_node = 8.5
    num_edges = int(num_nodes * avg_edges_per_node)
    
    # Input features: all prognostic variables at all levels
    # vorticity, divergence, temperature (3D) + log_surface_pressure (2D)
    input_dim = 3 * num_levels + 1
    
    # 1. Feature extraction (reshaping, normalization): ~10 FLOPs per node per feature
    feature_extraction_flops = 10 * num_nodes * input_dim
    
    # 2. GNN layers
    gnn_layer_flops = 0
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        gnn_layer_flops += estimate_gnn_layer_flops(num_nodes, num_edges, 
                                                     current_dim, hidden_dim)
        current_dim = hidden_dim
    
    # 3. Output layer
    output_dim = 3 * num_levels + 1  # same as input (corrections for each variable)
    output_layer_flops = estimate_gnn_layer_flops(num_nodes, num_edges,
                                                   current_dim, output_dim)
    
    # 4. Conversion to modal space (spectral transform per field)
    # This is a forward transform
    truncation = lon // 3  # approximate
    transform_flops = estimate_spectral_transform_flops(truncation, lon, lat)
    total_fields = 3 * num_levels + 1
    modal_conversion_flops = transform_flops * total_fields
    
    total = (feature_extraction_flops + gnn_layer_flops + 
             output_layer_flops + modal_conversion_flops)
    
    return total


def estimate_total_flops(grid_name: str) -> FLOPEstimate:
    """Estimate total FLOPs for one 6-hour timestep on given grid."""
    config = GRID_CONFIGS[grid_name]
    lon, lat, truncation = config["lon"], config["lat"], config["truncation"]
    
    # Number of steps
    num_dycore_steps = TOTAL_TIME_6H // DT_MODEL  # 60
    num_corrections = TOTAL_TIME_6H // CORRECTION_INTERVAL  # 6
    
    # Dycore FLOPs
    dycore_per_step = estimate_dycore_step_flops(lon, lat, truncation)
    dycore_total = dycore_per_step * num_dycore_steps
    
    # GNN FLOPs
    gnn_per_correction = estimate_gnn_correction_flops(lon, lat)
    gnn_total = gnn_per_correction * num_corrections
    
    total = dycore_total + gnn_total
    
    return FLOPEstimate(
        dycore_per_step=dycore_per_step,
        dycore_total=dycore_total,
        gnn_per_correction=gnn_per_correction,
        gnn_total=gnn_total,
        total=total
    )


def estimate_pure_gnn_flops(lon: int, lat: int, 
                             hidden_dims: Tuple[int, ...],
                             num_levels: int = NUM_VERTICAL_LEVELS) -> float:
    """Estimate FLOPs for a pure GNN model (single forward pass, no dycore)."""
    num_nodes = lon * lat
    avg_edges_per_node = 8.5
    num_edges = int(num_nodes * avg_edges_per_node)
    
    # Input: all prognostic variables
    input_dim = 3 * num_levels + 1
    
    # Feature extraction
    feature_extraction_flops = 10 * num_nodes * input_dim
    
    # GNN layers
    gnn_layer_flops = 0
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        gnn_layer_flops += estimate_gnn_layer_flops(num_nodes, num_edges, 
                                                     current_dim, hidden_dim)
        current_dim = hidden_dim
    
    # Output layer
    output_dim = 3 * num_levels + 1
    output_layer_flops = estimate_gnn_layer_flops(num_nodes, num_edges,
                                                   current_dim, output_dim)
    
    return feature_extraction_flops + gnn_layer_flops + output_layer_flops


def find_equivalent_gnn_size(target_flops: float, lon: int, lat: int,
                              num_levels: int = NUM_VERTICAL_LEVELS) -> Dict:
    """Find GNN configurations that approximately match the target FLOPs."""
    results = {}
    
    # Test different configurations
    # Format: (num_layers, hidden_dim)
    configs_to_test = []
    
    # Varying depth with fixed width
    for num_layers in range(2, 20):
        for hidden_dim in [64, 128, 256, 512, 1024]:
            configs_to_test.append((num_layers, hidden_dim))
    
    # Also test very wide networks
    for hidden_dim in [64, 128, 256, 512, 768, 1024, 1536, 2048]:
        for num_layers in [2, 4, 6, 8, 10, 12, 16]:
            configs_to_test.append((num_layers, hidden_dim))
    
    best_configs = []
    
    for num_layers, hidden_dim in configs_to_test:
        hidden_dims = tuple([hidden_dim] * num_layers)
        flops = estimate_pure_gnn_flops(lon, lat, hidden_dims, num_levels)
        
        # Check if within 20% of target
        if 0.8 * target_flops <= flops <= 1.2 * target_flops:
            params = estimate_gnn_params(lon, lat, hidden_dims, num_levels)
            best_configs.append({
                'num_layers': num_layers,
                'hidden_dim': hidden_dim,
                'flops': flops,
                'params': params,
                'ratio': flops / target_flops
            })
    
    # Sort by how close to target
    best_configs.sort(key=lambda x: abs(1 - x['ratio']))
    
    return best_configs[:10]  # Return top 10 matches


def estimate_gnn_params(lon: int, lat: int, 
                        hidden_dims: Tuple[int, ...],
                        num_levels: int = NUM_VERTICAL_LEVELS) -> int:
    """Estimate number of parameters in a GNN model."""
    input_dim = 3 * num_levels + 1
    output_dim = input_dim
    
    total_params = 0
    current_dim = input_dim
    
    for hidden_dim in hidden_dims:
        # Edge MLP: (edge_dim + 2*node_dim) -> hidden_dim
        edge_input = current_dim + 2 * current_dim
        total_params += edge_input * hidden_dim + hidden_dim  # weights + bias
        
        # Node MLP: (node_dim + hidden_dim) -> hidden_dim
        node_input = current_dim + hidden_dim
        total_params += node_input * hidden_dim + hidden_dim
        
        current_dim = hidden_dim
    
    # Output layer
    edge_input = current_dim + 2 * current_dim
    total_params += edge_input * output_dim + output_dim
    node_input = current_dim + output_dim
    total_params += node_input * output_dim + output_dim
    
    return total_params


def main():
    print("=" * 70)
    print("FLOP Estimates for Hybrid Model (6-hour timestep)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  dt_model = {DT_MODEL}s → {TOTAL_TIME_6H // DT_MODEL} dycore steps")
    print(f"  correction_interval = {CORRECTION_INTERVAL}s → {TOTAL_TIME_6H // CORRECTION_INTERVAL} GNN corrections")
    print(f"  GNN hidden_dims = {GNN_HIDDEN_DIMS}")
    print(f"  Vertical levels = {NUM_VERTICAL_LEVELS}")
    print()
    
    results = {}
    for grid_name in ["TL31", "TL47", "TL63", "TL95", "TL127"]:
        config = GRID_CONFIGS[grid_name]
        estimate = estimate_total_flops(grid_name)
        results[grid_name] = estimate
        
        print(f"\n{grid_name} ({config['lon']}×{config['lat']} = {config['lon']*config['lat']} nodes):")
        print(estimate)
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE - HYBRID MODEL")
    print("=" * 70)
    print(f"{'Grid':<8} {'Nodes':<10} {'Dycore':<15} {'GNN':<15} {'Total':<15} {'GFLOPs':<10}")
    print("-" * 70)
    for grid_name, est in results.items():
        config = GRID_CONFIGS[grid_name]
        nodes = config['lon'] * config['lat']
        gflops = est.total / 1e9
        print(f"{grid_name:<8} {nodes:<10} {est.dycore_total:.2e}  {est.gnn_total:.2e}  {est.total:.2e}  {gflops:.2f}")
    
    # Pure GNN equivalent analysis
    print("\n" + "=" * 70)
    print("EQUIVALENT PURE-GNN CONFIGURATIONS")
    print("(Same FLOP budget as hybrid, but no dycore)")
    print("=" * 70)
    
    for grid_name in ["TL31", "TL47", "TL63", "TL95", "TL127"]:
        config = GRID_CONFIGS[grid_name]
        hybrid_est = results[grid_name]
        target_flops = hybrid_est.total
        
        # Current small GNN for comparison
        current_gnn_flops = estimate_pure_gnn_flops(
            config['lon'], config['lat'], GNN_HIDDEN_DIMS)
        current_params = estimate_gnn_params(
            config['lon'], config['lat'], GNN_HIDDEN_DIMS)
        
        print(f"\n{grid_name} (target: {target_flops/1e9:.1f} GFLOPs):")
        print(f"  Current GNN (2×64): {current_gnn_flops/1e9:.2f} GFLOPs, {current_params:,} params")
        print(f"  Budget multiplier: {target_flops/current_gnn_flops:.1f}× larger GNN possible")
        
        # Find equivalent configs
        equiv_configs = find_equivalent_gnn_size(
            target_flops, config['lon'], config['lat'])
        
        if equiv_configs:
            print(f"  Equivalent pure-GNN architectures:")
            for i, cfg in enumerate(equiv_configs[:5]):
                print(f"    {i+1}. {cfg['num_layers']} layers × {cfg['hidden_dim']} hidden: "
                      f"{cfg['flops']/1e9:.1f} GFLOPs, {cfg['params']:,} params")
    
    # Comparison table
    print("\n" + "=" * 70)
    print("RECOMMENDED PURE-GNN SIZES (matching hybrid FLOP budget)")
    print("=" * 70)
    print(f"{'Grid':<8} {'Current':<12} {'Equivalent (deep)':<20} {'Equivalent (wide)':<20} {'Multiplier':<10}")
    print("-" * 70)
    
    for grid_name in ["TL31", "TL47", "TL63", "TL95", "TL127"]:
        config = GRID_CONFIGS[grid_name]
        hybrid_est = results[grid_name]
        target_flops = hybrid_est.total
        
        current_gnn_flops = estimate_pure_gnn_flops(
            config['lon'], config['lat'], GNN_HIDDEN_DIMS)
        
        equiv_configs = find_equivalent_gnn_size(
            target_flops, config['lon'], config['lat'])
        
        # Find deep (many layers) and wide (large hidden) options
        deep_cfg = None
        wide_cfg = None
        for cfg in equiv_configs:
            if cfg['num_layers'] >= 8 and deep_cfg is None:
                deep_cfg = cfg
            if cfg['hidden_dim'] >= 512 and wide_cfg is None:
                wide_cfg = cfg
        
        deep_str = f"{deep_cfg['num_layers']}×{deep_cfg['hidden_dim']}" if deep_cfg else "N/A"
        wide_str = f"{wide_cfg['num_layers']}×{wide_cfg['hidden_dim']}" if wide_cfg else "N/A"
        multiplier = target_flops / current_gnn_flops
        
        print(f"{grid_name:<8} 2×64        {deep_str:<20} {wide_str:<20} {multiplier:.0f}×")
    
    print("\n" + "=" * 70)
    print("Notes:")
    print("- These are theoretical estimates; actual FLOPs may vary ±50%")
    print("- Pure GNN would need to learn dynamics that dycore provides for free")
    print("- Consider: depth (more layers) vs width (larger hidden) trade-off")
    print("- Memory usage scales with hidden_dim² and num_layers")
    print("- For autoregressive rollout, multiply by number of steps")
    print("=" * 70)


if __name__ == "__main__":
    main()

