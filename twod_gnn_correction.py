"""GCN-based correction model for the hybrid dynamical core."""

from typing import Dict, List, Tuple, Union, Callable, Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from flax.core import freeze, unfreeze

from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import typing
from dinosaur import scales
from data_cacher import ERA5DataPreprocessor, setup_reference_temperature


def make_node_mlp(features):
    def update_node_fn(nodes, sent_edges, received_edges, globals_):
        inputs = jnp.concatenate([nodes, received_edges], axis=1)
        return jax.nn.relu(nn.Dense(features)(inputs))
    return update_node_fn
    
def make_edge_mlp(features):
    def update_edge_fn(edges, senders, receivers, globals_):
        inputs = jnp.concatenate([edges, senders, receivers], axis=1)
        return jax.nn.relu(nn.Dense(features)(inputs))
    return update_edge_fn


class GraphNetworkLayer(nn.Module):
    """Graph network layer for the GNN correction model using Jraph."""
    
    output_dim: int

    @nn.compact
    def __call__(
        self,
        graph: jraph.GraphsTuple,
    ) -> jraph.GraphsTuple:
        """Apply graph convolution using Jraph.
        
        Args:
            graph: GraphsTuple containing node features, edge indices, etc.
            
        Returns:
            Updated GraphsTuple with new node features.
        """
            
        # Create a GraphNetwork with our update functions
        gn = jraph.GraphNetwork(
            update_node_fn=make_node_mlp(self.output_dim),
            update_edge_fn=make_edge_mlp(self.output_dim),
            update_global_fn=None,
            aggregate_edges_for_nodes_fn=jraph.segment_sum,
            aggregate_nodes_for_globals_fn=None,
            aggregate_edges_for_globals_fn=None,
        )
        
        return gn(graph)

def make_output_node_mlp(features):
    def update_node_fn(nodes, sent_edges, received_edges, globals_):
        inputs = jnp.concatenate([nodes, received_edges], axis=1)
        return nn.Dense(features)(inputs)
    return update_node_fn
        
def make_output_edge_mlp(features):
    def update_edge_fn(edges, senders, receivers, globals_):
        inputs = jnp.concatenate([edges, senders, receivers], axis=1)
        return nn.Dense(features)(inputs)
    return update_edge_fn

class OutputLayer(nn.Module):
    """Output layer for GNN without activation function."""
    
    output_dim: int
    
    @nn.compact
    def __call__(
        self,
        graph: jraph.GraphsTuple,
    ) -> jraph.GraphsTuple:
        """Apply graph convolution using Jraph (without activation)."""
            
        # Create a GraphNetwork with our update functions
        gn = jraph.GraphNetwork(
            update_node_fn=make_output_node_mlp(self.output_dim),
            update_edge_fn=make_output_edge_mlp(self.output_dim),
            update_global_fn=None,
            aggregate_edges_for_nodes_fn=jraph.segment_sum,
            aggregate_nodes_for_globals_fn=None,
            aggregate_edges_for_globals_fn=None,
        )
        
        return gn(graph)


class GNNCorrection(nn.Module):
    """GNN-based correction model for the hybrid dynamical core."""
    
    coords: coordinate_systems.CoordinateSystem
    physics_specs: primitive_equations.PrimitiveEquationsSpecs
    factor: float
    temp_factor: float = 1
    hidden_dims: Tuple[int, ...] = (64, 64)
    
    def setup(self):
        # Define GNN layers
        self.gnn_layers = [GraphNetworkLayer(dim) for dim in self.hidden_dims]
        self.output_layer = OutputLayer(self._get_output_dim())

        senders, receivers, valid = self._construct_graph_numpy()

        # Create static variables - these won't be traced or differentiated
        self.senders = jnp.array(senders)
        self.receivers = jnp.array(receivers)
        self.valid = jnp.array(valid)

        self.num_nodes = self.coords.horizontal.nodal_shape[0] * self.coords.horizontal.nodal_shape[1]
        self.num_edges = self.senders.shape[0]
    
    def __call__(
        self,
        state: typing.PyTreeState,
    ) -> typing.PyTreeState:
        """Compute correction for the given state.
        
        Args:
            state: Current state in modal space.
            
        Returns:
            Correction in modal space.
        """

        # Extract node features from the nodal state
        node_features, stdevs = self._extract_node_features(state)

        senders = jax.lax.stop_gradient(self.senders)
        receivers = jax.lax.stop_gradient(self.receivers)
        valid = jax.lax.stop_gradient(self.valid)

        # Create edge features (ones with same feature dimension as nodes)
        all_edges = jnp.ones((self.num_edges, node_features.shape[1]))
        actual_edges = all_edges * jnp.expand_dims(valid, axis=-1)
        
        # Create GraphsTuple
        graph = jraph.GraphsTuple(
            nodes=node_features,
            edges=actual_edges,
            senders=senders,
            receivers=receivers,
            globals=None,
            n_node=jnp.array([self.num_nodes]),
            n_edge=jnp.array([self.num_edges])
        )
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            graph = layer(graph)
        
        # Final layer to predict corrections for each variable
        graph = self.output_layer(graph)
        
        
        # Reshape and convert back to modal space
        correction_features = graph.nodes
        nodal_correction = self._reshape_correction(correction_features, stdevs, self.factor)
        modal_correction = self._convert_to_modal(nodal_correction)
        
        return modal_correction

    def _construct_graph_numpy(self):
        """Construct graph using NumPy to ensure it's static to JAX."""
        # Get grid dimensions
        lon_size, lat_size = self.coords.horizontal.nodal_shape
        num_nodes = lon_size * lat_size
        
        # Create meshgrid
        i_indices, j_indices = np.meshgrid(np.arange(lon_size), np.arange(lat_size), indexing='ij')
        i_indices = i_indices.reshape(-1)
        j_indices = j_indices.reshape(-1)
        
        # Define offsets for 9-point stencil
        di_offsets = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        dj_offsets = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        
        # Pre-allocate arrays for maximum possible edges
        max_edges = num_nodes * 9
        senders = np.zeros(max_edges, dtype=np.int32)
        receivers = np.zeros(max_edges, dtype=np.int32)
        valid = np.zeros(max_edges, dtype=np.bool_)
        
        # Function to compute linear index
        def linear_index(i, j):
            return i * lat_size + j
        
        # Track the actual number of valid edges
        edge_count = 0
        
        # Process all nodes
        for node_idx in range(num_nodes):
            node_i, node_j = i_indices[node_idx], j_indices[node_idx]
            node_linear_idx = linear_index(node_i, node_j)
            
            # Process all 9 stencil points
            for k in range(9):
                ni = (node_i + di_offsets[k]) % lon_size  # Apply longitude periodicity
                nj = node_j + dj_offsets[k]
                
                # Check if neighbor is valid
                is_valid = (0 <= nj < lat_size)
                
                if is_valid:
                    # Add valid edge
                    senders[edge_count] = node_linear_idx
                    receivers[edge_count] = linear_index(ni, nj)
                    valid[edge_count] = True
                    edge_count += 1
        
        # Trim arrays to actual size
        senders = senders[:edge_count]
        receivers = receivers[:edge_count]
        valid = valid[:edge_count]
        
        return senders, receivers, valid
    
    def construct_graph_jit_compatible(self):
        """Graph construction that works with JIT."""
        # Get grid dimensions
        lon_size, lat_size = self.coords.horizontal.nodal_shape
        num_nodes = lon_size * lat_size
        max_edges = num_nodes * 9  # Maximum possible edges (9-point stencil)
        
        # Create meshgrid
        i_indices, j_indices = np.meshgrid(np.arange(lon_size), np.arange(lat_size), indexing='ij')
        i_indices = i_indices.reshape(-1)
        j_indices = j_indices.reshape(-1)
        
        # Define offsets for 9-point stencil
        di_offsets = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        dj_offsets = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        
        # Pre-allocate arrays of fixed size
        senders = np.zeros(max_edges, dtype=np.int32)
        receivers = np.zeros(max_edges, dtype=np.int32)
        valid = np.zeros(max_edges, dtype=np.bool_)
        
        # Function to compute linear index
        def linear_index(i, j):
            return i * lat_size + j
        
        # Process all nodes and stencil offsets
        def process_node(idx, arrays):
            senders, receivers, valid = arrays
            node_i, node_j = i_indices[idx], j_indices[idx]
            node_idx = linear_index(node_i, node_j)
            
            # Base offset for this node's edges
            base_offset = idx * 9
            
            # Process all 9 stencil points
            for k in range(9):
                ni = (node_i + di_offsets[k]) % lon_size  # Apply longitude periodicity
                nj = node_j + dj_offsets[k]
                
                # Check if neighbor is valid
                is_valid = (nj >= 0) & (nj < lat_size)
                
                # Calculate edge index
                edge_idx = base_offset + k
                
                # Set values using dynamic_update_slice
                senders[edge_idx] = np.where(is_valid, node_idx, 0)
                receivers[edge_idx] = np.where(is_valid, linear_index(ni, nj), 0)
                valid[edge_idx] = is_valid
                
            return senders, receivers, valid
        
        # Scan over all nodes with for loop
        for idx in range(num_nodes):
            senders[idx], receivers[idx], valid[idx] = process_node(idx, (senders, receivers, valid))
        
        # Outputs that work with GraphsTuple
        return senders, receivers, valid

    def _convert_to_modal(self, nodal_correction: Dict) -> typing.PyTreeState:
        """Convert correction from nodal to modal space.
        
        Args:
            nodal_correction: Correction in nodal space.
            
        Returns:
            Correction in modal space.
        """
        modal_dict = {}
        
        # Process u and v wind components to get vorticity and divergence
        if 'u_component_of_wind' in nodal_correction and 'v_component_of_wind' in nodal_correction:
            
            u_nodal = nodal_correction['u_component_of_wind']
            v_nodal = nodal_correction['v_component_of_wind']
            
            vorticity, divergence = ERA5DataPreprocessor._uv_nodal_to_vor_div_modal(self.coords.horizontal, u_nodal, v_nodal)

            modal_dict['vorticity'] = vorticity
            modal_dict['divergence'] = divergence
        
        # Handle temperature - convert to temperature variation if present
        if 'temperature_variation' in nodal_correction:
            modal_dict['temperature_variation'] = self.coords.horizontal.to_modal(
                nodal_correction['temperature_variation']
            )
        
        # Handle log surface pressure correction
        if 'log_surface_pressure' in nodal_correction:
            # Convert correction directly as log surface pressure correction
            modal_pressure = self.coords.horizontal.to_modal(
                nodal_correction['log_surface_pressure']
            )
            modal_dict['log_surface_pressure'] = modal_pressure[jnp.newaxis, :, :]
        
        # Handle specific humidity
        if 'tracers' in nodal_correction and 'specific_humidity' in nodal_correction['tracers']:
            modal_dict['tracers'] = {'specific_humidity': self.coords.horizontal.to_modal(
                nodal_correction['tracers']['specific_humidity']
            )}
        
        modal_dict['sim_time'] = 0

        # Convert back to the appropriate state type
        return primitive_equations.State(**modal_dict)
    
    def _extract_node_features(self, nodal_state: Dict) -> jnp.ndarray:
        """Extract node features from the nodal state.
        
        Args:
            nodal_state: State in nodal space.
            
        Returns:
            Node features of shape [num_nodes, feature_dim].
            Stdevs of shape [feature_dim].
        """
        # Extract relevant variables
        features = []
        stdevs = []
        
        #add u component of wind
        if 'u_component_of_wind' in nodal_state and nodal_state['u_component_of_wind'] is not None:
            u_wind = nodal_state['u_component_of_wind']
            # Reshape to [num_nodes, num_levels]
            u_wind_flat = u_wind.reshape(-1, u_wind.shape[0])
            features.append(u_wind_flat)
            stdevs.append(jnp.repeat(jnp.std(u_wind_flat), u_wind_flat.shape[1]))

        #add v component of wind
        if 'v_component_of_wind' in nodal_state and nodal_state['v_component_of_wind'] is not None:
            v_wind = nodal_state['v_component_of_wind']
            # Reshape to [num_nodes, num_levels]
            v_wind_flat = v_wind.reshape(-1, v_wind.shape[0])
            features.append(v_wind_flat)
            stdevs.append(jnp.repeat(jnp.std(v_wind_flat), v_wind_flat.shape[1]))
        # Add vorticity
        if 'vorticity' in nodal_state and nodal_state['vorticity'] is not None:
            vorticity = nodal_state['vorticity']
            # Reshape to [num_nodes, num_levels]
            vorticity_flat = vorticity.reshape(-1, vorticity.shape[0])
            features.append(vorticity_flat)
            stdevs.append(jnp.repeat(jnp.std(vorticity_flat), vorticity_flat.shape[1]  ))
        # Add divergence
        if 'divergence' in nodal_state and nodal_state['divergence'] is not None:
            divergence = nodal_state['divergence']
            # Reshape to [num_nodes, num_levels]
            divergence_flat = divergence.reshape(-1, divergence.shape[0])
            features.append(divergence_flat)
            stdevs.append(jnp.repeat(jnp.std(divergence_flat), divergence_flat.shape[1]))

        # Add temperature variation
        if 'temperature_variation' in nodal_state and nodal_state['temperature_variation'] is not None:
            temp = nodal_state['temperature_variation']
            # Reshape to [num_nodes, num_levels]
            temp_flat = temp.reshape(-1, temp.shape[0])
            features.append(temp_flat)
            stdevs.append(jnp.repeat(jnp.std(temp_flat) * self.temp_factor, temp_flat.shape[1]))

        # Add log surface pressure
        if 'log_surface_pressure' in nodal_state and nodal_state['log_surface_pressure'] is not None:
            sp = nodal_state['log_surface_pressure']
            # Reshape to [num_nodes, 1]
            sp_flat = sp.reshape(-1, 1)
            features.append(sp_flat)
            stdevs.append(jnp.repeat(jnp.std(sp_flat), sp_flat.shape[1]))

        # Add specific humidity
        if 'tracers' in nodal_state and nodal_state['tracers'] is not None:
            specific_humidity = nodal_state['tracers']['specific_humidity']
            # Reshape to [num_nodes, num_levels]
            specific_humidity_flat = specific_humidity.reshape(-1, specific_humidity.shape[0])
            features.append(specific_humidity_flat)
            stdevs.append(jnp.repeat(jnp.std(specific_humidity_flat), specific_humidity_flat.shape[1]))
        
            
        # Concatenate all features
        node_features = jnp.concatenate(features, axis=1)
        stdevs = jnp.concatenate(stdevs, axis=0)
        return node_features, stdevs
    
    def _construct_graph(self) -> np.ndarray:
        """Construct GraphsTuple based on the grid using JAX vectorization.
        
        Args:
            node_features: Node features of shape [num_nodes, feature_dim].
            
        Returns:
            GraphsTuple representation of the grid.
        """
        # Get grid dimensions
        lon_size, lat_size = self.coords.horizontal.nodal_shape
        num_nodes = lon_size * lat_size
        
        # Create a coordinate grid for all nodes
        i_indices, j_indices = np.meshgrid(np.arange(lon_size), np.arange(lat_size), indexing='ij')
        i_indices = i_indices.reshape(-1)  # Flatten to a 1D array of i coordinates
        j_indices = j_indices.reshape(-1)  # Flatten to a 1D array of j coordinates
        
        # Define offsets for the 9-point stencil (including self)
        di_offsets = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        dj_offsets = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        num_offsets = len(di_offsets)
        
        # Function to compute linear index from i, j coordinates
        def linear_index(i, j):
            return i * lat_size + j
        
        # Create vectorized version of the linear_index function
        vlinear_index = np.vectorize(linear_index)
        
        # Vectorize the creation of sender and receiver indices
        def process_node(node_i, node_j):
            # Compute offsets for this node
            ni = (node_i + di_offsets) % lon_size  # Apply longitude periodicity
            nj = node_j + dj_offsets
            
            # Create mask for valid latitude indices
            valid_mask = (nj >= 0) & (nj < lat_size)
            
            # Create sender and receiver indices
            node_idx = linear_index(node_i, node_j)
            senders = np.full(num_offsets, node_idx)
            receivers = vlinear_index(ni, nj)
            
            # Apply mask
            valid_senders = senders * valid_mask
            valid_receivers = receivers * valid_mask
            
            return valid_senders, valid_receivers, valid_mask
        
        # Vectorize the process_node function over all nodes
        vprocess_node = np.vectorize(process_node)
        
        # Process all nodes to get all senders and receivers
        all_senders, all_receivers, valid_mask = vprocess_node(i_indices, j_indices)
        
        # Flatten and filter valid connections
        flat_senders = all_senders.reshape(-1)
        flat_receivers = all_receivers.reshape(-1)
        flat_mask = valid_mask.reshape(-1)
        
        # Filter out invalid connections
        valid_connections = flat_mask == 1
        senders = flat_senders[valid_connections]
        receivers = flat_receivers[valid_connections]
        
        return senders, receivers, num_nodes
    
    def _get_output_dim(self) -> int:
        """Get the output dimension of the GNN."""
        output_dim = 0
        output_dim += 6 * self.coords.vertical.layers # u, v, vorticity, divergence, temperature, specific humidity
        output_dim += 1 # log surface pressure
        return output_dim
    
    def _reshape_correction(self, correction_features: jnp.ndarray, stdevs: jnp.ndarray, factor: float) -> Dict:
        """Reshape correction features to match state variables.
        
        Args:
            correction_features: Correction features of shape [num_nodes, output_dim].
            
        Returns:
            Dictionary of correction variables in nodal space.
        """
        # Get grid dimensions
        lon_size, lat_size = self.coords.horizontal.nodal_shape
        num_levels = self.coords.vertical.layers
        
        # correction_features: [num_nodes, feature_dim], stdevs: [feature_dim]
        norm_factors = stdevs * factor
        correction_features = correction_features * norm_factors[jnp.newaxis, :]

        # Split correction features
        start_idx = 0

        # U wind correction
        u_wind_size = num_levels
        u_wind_correction = correction_features[:, start_idx:start_idx+u_wind_size]
        u_wind_correction = u_wind_correction.reshape(lon_size, lat_size, num_levels)
        u_wind_correction = jnp.transpose(u_wind_correction, (2, 0, 1)) # [levels, lon, lat]
        
        start_idx += u_wind_size

        # V wind correction
        v_wind_size = num_levels
        v_wind_correction = correction_features[:, start_idx:start_idx+v_wind_size]
        v_wind_correction = v_wind_correction.reshape(lon_size, lat_size, num_levels)
        v_wind_correction = jnp.transpose(v_wind_correction, (2, 0, 1))  # [levels, lon, lat]
        start_idx += v_wind_size
        
        # Vorticity correction
        vorticity_size = num_levels
        vorticity_correction = correction_features[:, start_idx:start_idx+vorticity_size]
        vorticity_correction = vorticity_correction.reshape(lon_size, lat_size, num_levels)
        vorticity_correction = jnp.transpose(vorticity_correction, (2, 0, 1)) 
        start_idx += vorticity_size
        
        # Divergence correction
        divergence_size = num_levels
        divergence_correction = correction_features[:, start_idx:start_idx+divergence_size]
        divergence_correction = divergence_correction.reshape(lon_size, lat_size, num_levels)
        divergence_correction = jnp.transpose(divergence_correction, (2, 0, 1))  # [levels, lon, lat]
        start_idx += divergence_size
        
        # Temperature correction
        temp_size = num_levels
        temp_correction = correction_features[:, start_idx:start_idx+temp_size]
        temp_correction = temp_correction.reshape(lon_size, lat_size, num_levels)
        temp_correction = jnp.transpose(temp_correction, (2, 0, 1)) 
        start_idx += temp_size
        
        # Log surface pressure correction
        sp_correction = correction_features[:, start_idx:start_idx+1]
        sp_correction = sp_correction.reshape(lon_size, lat_size)

        # Specific humidity correction
        specific_humidity_size = num_levels
        specific_humidity_correction = correction_features[:, start_idx+1:start_idx+1+specific_humidity_size]
        specific_humidity_correction = specific_humidity_correction.reshape(lon_size, lat_size, num_levels)
        specific_humidity_correction = jnp.transpose(specific_humidity_correction, (2, 0, 1))  # [levels, lon, lat]
        
        # Create correction dictionary
        correction = {
            'u_component_of_wind': u_wind_correction,
            'v_component_of_wind': v_wind_correction,
            'vorticity': vorticity_correction,
            'divergence': divergence_correction,
            'temperature_variation': temp_correction,
            'log_surface_pressure': sp_correction,
            'tracers': {'specific_humidity': specific_humidity_correction}
        }

        return correction
    
class NullCorrection(nn.Module):
    """A stub correction module that always returns zero."""
    @nn.compact
    def __call__(self, x: primitive_equations.State) -> primitive_equations.State:
        # Keep dummy parameter but generate zero via dynamic subtraction to avoid literal zeros
        dummy = self.param("dummy", nn.initializers.zeros, ())
        zero = dummy - dummy
        # Create a zero PyTree matching x's structure using dynamic zero
        return jax.tree.map(lambda v: jnp.full(v.shape, zero, dtype=v.dtype), x)