"""Training script for the hybrid model."""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

import google.auth.compute_engine._metadata as _md
_md._NUM_METADATA_SERVER_RETRIES = 1         # instead of 5
_md._METADATA_DEFAULT_TIMEOUT = 0     # belt-and-braces

import argparse
import os
from typing import Dict, Tuple, Any, List
import time
from datetime import datetime, timedelta

import jax
import jax.numpy as jnp
import flax
import numpy as np
import optax
import orbax.checkpoint
import xarray as xr
import zarr
import wandb
from flax.training import train_state
from flax.training.dynamic_scale import DynamicScale
from tqdm import tqdm

from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur.spherical_harmonic import FastSphericalHarmonics
from dinosaur import time_integration
from dinosaur import typing

from gcm import HybridModel
from dycore import DynamicalCore
from data_utils import load_era5_data
from dataloader import create_dataloader
from utils import train_pmap, valid_pmap, unreplicate_metrics
from utils import shard, aggregate_metrics, compute_state_mse, shard_trajectory
from utils import test_pmap_fn, metrics_process_fn
from utils import xarray_process_fn
from utils import get_start_time, get_end_time
from utils import finalize_forecast_zarr, save_dataset_to_zarr, trajectory_to_xarray
from utils import save_gain_to_zarr
CHKPT_DIR = os.environ.get("CHKPT_DIR", os.path.join(os.path.dirname(__file__), "checkpoints"))
jax.config.update("jax_enable_x64", False)
#jax.config.update("jax_default_matmul_precision", "float16")



def init_and_train_model(
    model: HybridModel,
    train_loader: Any,
    train_loader_args: Any,
    valid_loader: Any,
    train_length: int,
    valid_interval: int,
    num_epochs: int,
    learning_rate: float,
    log_wandb: bool = True,
    restore_checkpoint_manager: orbax.checkpoint.CheckpointManager = None,
    save_checkpoint_manager: orbax.checkpoint.CheckpointManager = None,
    load_checkpoint: bool = False,
    correction: str = "GNNCorrection",
    seed: int = 42,
) -> Dict:
    """Train the hybrid model.
    
    This function coordinates the entire training process:
    1. Initializes model parameters using an example state
    2. Sets up the optimizer
    3. Runs the training loop for the specified number of epochs
    4. For each epoch, iterates through all training samples
    5. Logs training metrics and saves model checkpoints
    
    The data_loader provides pairs of (input_data, target_data) where
    target_data is prediction_range timesteps ahead of input_data in the
    original ERA5 dataset. The model is then trained to predict this
    target_data by running prediction_steps internal model steps.
    
    Args:
        model: Hybrid model.
        data_loader: Data loader for ERA5 data.
        num_epochs: Number of epochs.
        learning_rate: Learning rate.
        log_wandb: Whether to log metrics to wandb.
        load_checkpoint: Whether to load a model checkpoint.
    Returns:
        Trained model parameters.
    """
    time_start = time.perf_counter()
    devices = jax.device_count()
    print(f"Training on {devices} devices")

  
    first_batch = next(iter(train_loader))
    example_state = first_batch[0][0]
    # Initialize parameters
    rng_key = jax.random.PRNGKey(seed)
    variables = model.init(rng_key, example_state)

    print("Parameters initialised")

    learning_rate = learning_rate * devices
    optimizer = optax.adam(learning_rate)

    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.zero_nans(), optimizer)
    print("Using AdamW optimizer with clip")

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        tx=tx,
        params=variables["params"]
    )

    if load_checkpoint and restore_checkpoint_manager is not None:
        step_to_restore = restore_checkpoint_manager.latest_step()
        print(f"Loading parameters from checkpoint (latest step={step_to_restore})")
        if step_to_restore is None:
            print("WARNING: No checkpoint step found to restore from. Proceeding with fresh init.")
        else:
            state = restore_checkpoint_manager.restore(step_to_restore, items={"model": state})["model"]

    state = flax.jax_utils.replicate(state, devices=jax.devices())
    print("Train state created")

    best_state = flax.jax_utils.unreplicate(state)

    # Ensure checkpoint step numbers are monotonically increasing across runs
    save_step_base = 0
    if save_checkpoint_manager is not None:
        prev_latest = save_checkpoint_manager.latest_step()
        if prev_latest is not None:
            save_step_base = prev_latest
        print(f"Checkpoint save base step set to {save_step_base}")

    best_train_loss = float('inf')
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        datapoint_count = 1 * devices
        for batch_train_input, batch_train_target in train_loader:
            # Update parameters for each example in the batch
            train_inputs = shard(batch_train_input, devices)
            train_targets = shard_trajectory(batch_train_target, devices)
            print(f"Training step {datapoint_count}/{len(train_loader)*devices}")

            state, loss, (tracer_mse, vars_mse) = train_pmap(
                state=state,
                inputs=train_inputs,
                targets=train_targets,
            )
            loss = unreplicate_metrics(loss)
            tracer_mse = unreplicate_metrics(tracer_mse)
            vars_mse = unreplicate_metrics(vars_mse)
            if log_wandb:
                wandb.log({
                    "train/loss": float(loss),
                    **{f"train/{k}": float(v) for k, v in vars_mse.items()},
                    "train_step": datapoint_count,
                })
            #jax.profiler.stop_trace()
            datapoint_count += 1 * devices

            if datapoint_count % (valid_interval * devices) == 0:

                valid_step = 1 * devices

                valid_mse_aggr = []
                valid_losses = []
                for batch_valid_input, batch_valid_target in valid_loader:
                    valid_inputs = shard(batch_valid_input, devices)
                    valid_targets = shard_trajectory(batch_valid_target, devices)
                    valid_loss, (tracers_mse, valid_mse) = valid_pmap(
                        state=state,
                        inputs=valid_inputs,
                        targets=valid_targets
                    )
                    valid_loss = unreplicate_metrics(valid_loss)
                    valid_mse = unreplicate_metrics(valid_mse)
                    valid_losses.append(valid_loss)
                    valid_mse_aggr.append(valid_mse)
                    print(f"Validation step {valid_step}/{len(valid_loader)*devices}")
                    valid_step += 1 * devices

                valid_mse_aggr = aggregate_metrics(valid_mse_aggr)
                valid_loss_aggr = jnp.mean(jnp.asarray(valid_losses))
                
                #jax.profiler.stop_trace()

                if log_wandb:
                    wandb.log({
                        "valid/loss": float(valid_loss_aggr),
                        **{f"valid/{k}": float(v) for k, v in valid_mse_aggr.items()},
                        "valid_step": datapoint_count
                    })

            #jax.profiler.stop_trace()
            if not jnp.isnan(loss) and loss < best_train_loss:
                best_train_loss = loss
                best_state = flax.jax_utils.unreplicate(state)
                if save_checkpoint_manager is not None:
                    save_step = save_step_base + datapoint_count
                    save_checkpoint_manager.save(save_step, items={"model": best_state})
                    save_checkpoint_manager.wait_until_finished()

            elif jnp.isnan(loss) and datapoint_count > 30*devices:
                print("NaN loss detected, restoring best state")
                state = flax.jax_utils.replicate(best_state)
                continue
            elif jnp.isnan(loss) and datapoint_count < 20*devices:
                print("NaN loss detected, restoring best state")
                state = flax.jax_utils.replicate(best_state)
                continue
            elif jnp.isnan(loss) and 20*devices <= datapoint_count <= 30*devices:
                print("Model is not stable, stopping training")
                return "Restart with lower prediction range"

            if datapoint_count > train_length:
                return "Training complete, move to higher prediction range"
    
    state = flax.jax_utils.unreplicate(state)
    time_end = time.perf_counter()
    print(f"Training complete in {time_end - time_start} seconds")
    return "Training complete"

def test_model(
            model: HybridModel,
            xarray: bool,
            results_path: str,
            results_id: str,
            test_year: str,
            test_loader: Any,
            prediction_range: int,
            checkpoint_manager: orbax.checkpoint.CheckpointManager = None,
            time_values: List[str] = None,
            learning_rate: float = 1e-3,
            save_trajectories: bool = False,
            save_gain: bool = False,
    ):
    devices = jax.device_count()
    print(f"Testing on {devices} devices")
    time_start = time.perf_counter()

    if not os.path.exists(results_path):
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        wandb_id = wandb.run.id
        name = f"test_{test_year}_{model.prediction_range}_{wandb_id}_{results_id}.zarr"
        os.makedirs(f"{results_path}/{name}")
        results_path = f"{results_path}/{name}"
    else:
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        wandb_id = wandb.run.id
        results_path = f"{results_path}/test_{test_year}_{model.prediction_range}_{time_str}_{wandb_id}_{results_id}.zarr"

  
    first_batch = next(iter(test_loader))
    example_state = first_batch[0][0]
    # Initialize parameters
    rng_key = jax.random.PRNGKey(42)
    variables = model.init(rng_key, example_state)

    print("Parameters initialised for testing")

    optimizer = optax.adam(learning_rate)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optax.chain(optax.clip_by_global_norm(1.0), optax.zero_nans(), optimizer),
        params=variables["params"]
    )

    if checkpoint_manager is not None:
        print(f"Loading parameters from checkpoint")
        state = checkpoint_manager.restore(checkpoint_manager.best_step(), items={"model": state})["model"]
    else:
        raise ValueError("No checkpoint provided")

    state = flax.jax_utils.replicate(state, devices=jax.devices())
    print("Train state created")

    datapoint_count = 1 * devices

    test_losses = []
    tracer_mses = []
    vars_mses = []

    for batch_test_input, batch_test_target in tqdm(test_loader):

        time_value = time_values[datapoint_count - devices : datapoint_count]
        pred_time = time_values[(datapoint_count - devices) + prediction_range : datapoint_count + prediction_range]
        test_inputs = shard(batch_test_input, devices)
        test_targets = shard(batch_test_target, devices)
        print(f"Testing step {datapoint_count}/{len(test_loader)*devices}")

        step_start = time.perf_counter()

        output, aux1, aux2 = test_pmap_fn(
            state=state,
            inputs=test_inputs,
            targets=test_targets,
            xarray=xarray,
            dycore=model.dycore,
        )
        aux1, aux2 = jax.device_get((aux1, aux2))
        
        print(f"Inference time: {time.perf_counter() - step_start} seconds")
        
        output = metrics_process_fn(output, test_losses, tracer_mses, vars_mses, xarray)
        datapoint_count += 1 * devices

        if xarray:
            output = xarray_process_fn(output, model.dycore, np.array(time_value))

            save_dataset_to_zarr(output, results_path)
        
        if save_trajectories:
            dycore_output = trajectory_to_xarray(
                aux1,
                np.array(time_value),
                model.dycore,
            )

            save_dataset_to_zarr(dycore_output, results_path.replace('.zarr', '_dycore.zarr'))

            correction_output = trajectory_to_xarray(
                aux2,
                np.array(time_value),
                model.dycore,
            )

            save_dataset_to_zarr(correction_output, results_path.replace('.zarr', '_correction.zarr'))
            print(f"Saved trajectories {datapoint_count - devices}-{datapoint_count} to xarray datasets")
        
        if save_gain:
            name = results_path.replace('.zarr', '_gain.zarr')
            print("Pred time:", pred_time)
            print("Time value:", time_value)
            save_gain_to_zarr(aux1, np.array(time_value), np.array(pred_time), name)


    if xarray:

        out_path = finalize_forecast_zarr(
            results_path,
            model.dycore,
            prediction_range,
        )
        print(f"Finalized forecast zarr store at {out_path}")

        if save_trajectories:
            dycore_out_path = finalize_forecast_zarr(
                results_path.replace('.zarr', '_dycore.zarr'),
                model.dycore,
                prediction_range,
            )
            print(f"Finalized dycore zarr store at {dycore_out_path}")

            correction_out_path = finalize_forecast_zarr(
                results_path.replace('.zarr', '_correction.zarr'),
                model.dycore,
                prediction_range,
            )
            print(f"Finalized correction zarr store at {correction_out_path}")

    if not xarray:
        test_losses, tracer_mses, vars_mses = output
        test_loss_aggr = jnp.mean(jnp.asarray(test_losses))
        test_mse_aggr = aggregate_metrics(vars_mses)
        wandb.log({
            "test/loss": float(test_loss_aggr),
            **{f"test/{k}": float(v) for k, v in test_mse_aggr.items()},
                    "test_step": datapoint_count,
                })
    print(f"Testing complete in {time.perf_counter() - time_start} seconds, see results in {out_path}")


def main():
    """Main function.
    This function sets up the experiment, including:
    1. Parsing command-line arguments
    2. Setting up the coordinate system and physics
    3. Creating the dynamical core and hybrid model
    4. Creating the data loader
    5. Training the model
    6. Saving the final model
    
    The data loading flow is:
    - start_time and end_time define the overall dataset time range
    - The dataloader extracts all available timesteps in this range
    - prediction_range determines how many timesteps to skip for targets
    - For each training step, two timepoints are loaded: input and target
    - prediction_steps determines how many model steps to run to reach target
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train hybrid model")

    parser.add_argument("--data_path", type=str, default="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr", help="Path to ERA5 data")
    parser.add_argument("--save_path", type=str, default='./checkpoints', help="Path to save model")
    parser.add_argument("--train_length", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("CACHE_DIR", "./cache"), help="Directory from/to which processed data is cached")
    parser.add_argument("--use_modal_cache", type=bool, default=True, help="Use modal-space cached data")
    parser.add_argument("--test", action="store_true", help="Test running testing")
    parser.add_argument("--xarray", action="store_true", help="Test running testing")
    parser.add_argument("--results_path", type=str, default=os.environ.get("RESULTS_DIR", "./results"), help="Path to save results")
    parser.add_argument("--results_id", type=str, default="", help="ID of the results")
    parser.add_argument("--test_year", type=str, default=None, help="Year of the test data")
    parser.add_argument("--save_gain", action="store_true", help="Whether to save the gain")

    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="hybrid-weather", help="Wandb project name")
    parser.add_argument("--checkpoint_name", type=str, default=None, help="Base checkpoint name (e.g. experiment id)")
    parser.add_argument("--load_checkpoint", type=bool, default=False, help="Load model checkpoint")
    # New: allow separate restore/save subdirectories under base checkpoint name
    parser.add_argument("--load_checkpoint_name", type=str, default=None, help="Subdirectory to restore from (e.g. pr1)")
    parser.add_argument("--save_checkpoint_name", type=str, default=None, help="Subdirectory to save to (e.g. pr2)")
    

    # NEW: seed for reproducibility / variability across restarts
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data shuffling and parameter init")
    parser.add_argument("--valid_interval", type=int, default=10000, help="Interval between validations")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")

    parser.add_argument("--dt", type=float, default=360, help="Dynamical model time step in seconds")
    parser.add_argument("--dt_model", type=int, default=360, help="Model time step in seconds")
    parser.add_argument("--start_time", type=str, default="2020-01-01", help="Start time of the dataset time range")
    parser.add_argument("--end_time", type=str, default="2020-01-03", help="End time of the dataset time range")
    parser.add_argument('--valid_start', type=str, default='2020-01-06', help='The start of the interval on which the model is validated on')
    parser.add_argument('--valid_end', type=str, default='2020-01-31', help='The end of the interval on which the model is validated on')
    parser.add_argument("--prediction_range", type=int, default=40, help="The number of 6 hour timesteps being simulated to eg: 1 = 6 hours from input time, 2 = 12 hours")
    parser.add_argument("--correction_interval", type=int, default=3600, help="Time interval between corrections in seconds")
    parser.add_argument("--trajectory", action="store_true", help="Whether to run the model in trajectory mode")
    parser.add_argument("--save_trajectories", action="store_true", help="Whether to save the trajectories")

    parser.add_argument("--resolution", type=str, default="TL127", help="Model resolution (TL63, TL127, TL255)")
    parser.add_argument("--num_levels", type=int, default=13, help="Number of vertical levels")
    parser.add_argument("--correction", type=str, default="GNNCorrection", help="Wether to apply neural corrections: GNNCorrection, NullCorrection, FullTendency")
    parser.add_argument("--gnn_hidden_dims", type=int, nargs="+", default=(64, 64), help="Number of hidden dimensions for the GNN correction model")
    parser.add_argument("--dfi_timescale", type=float, default=21600.0, help="Digital filter initialization timescale in seconds (default: 6 hours)")
    args = parser.parse_args()
    
    print(f"Training model with resolution: {args.resolution} on {jax.device_count()} devices, trajectory: {'Yes' if args.trajectory else 'No'}")
    
    # Set up coordinate system
    if args.resolution == "TL127":
        horizontal_grid = spherical_harmonic.Grid.TL127()#spherical_harmonics_impl=FastSphericalHarmonics)
        print('Modal shape:', horizontal_grid.spherical_harmonics.modal_shape)
        print('Nodal shape:', horizontal_grid.spherical_harmonics.nodal_shape)
    elif args.resolution == "TL31":
        horizontal_grid = spherical_harmonic.Grid.TL31()
        print('Modal shape:', horizontal_grid.spherical_harmonics.modal_shape)
        print('Nodal shape:', horizontal_grid.spherical_harmonics.nodal_shape)
    elif args.resolution == "TL47":
        horizontal_grid = spherical_harmonic.Grid.TL47()
        print('Modal shape:', horizontal_grid.spherical_harmonics.modal_shape)
        print('Nodal shape:', horizontal_grid.spherical_harmonics.nodal_shape)
    elif args.resolution == "TL63":
        horizontal_grid = spherical_harmonic.Grid.TL63()
        print('Modal shape:', horizontal_grid.spherical_harmonics.modal_shape)
        print('Nodal shape:', horizontal_grid.spherical_harmonics.nodal_shape)
    elif args.resolution == "TL95":
        horizontal_grid = spherical_harmonic.Grid.TL95()
        print('Modal shape:', horizontal_grid.spherical_harmonics.modal_shape)
        print('Nodal shape:', horizontal_grid.spherical_harmonics.nodal_shape)

    
    if args.batch_size != jax.device_count():
        raise ValueError(f"Batch size must be equal to the number of devices, got {args.batch_size} != {jax.device_count()}")

    if args.save_trajectories and not args.test:
        raise ValueError('Trajectory saving is only supported in test mode')

    if args.save_gain and not args.test:
        raise ValueError('Gain saving is only supported in test mode')
    
    if args.save_trajectories and args.save_gain:
        raise ValueError('Trajectory and gain saving cannot be enabled at the same time')
    
    vertical_grid = sigma_coordinates.SigmaCoordinates.equidistant(args.num_levels)
    coords = coordinate_systems.CoordinateSystem(horizontal_grid, vertical_grid)
    print(f"Coordinate system: {coords}")

    # Set up physics specifications
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    print(f"Physics specs: {physics_specs}")
    
    # Create train loader
    print(f"Creating train loader with time range from {args.start_time} to {args.end_time}")
    train_loader_args = {
        "data_path": args.data_path,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "coords": coords,
        "physics_specs": physics_specs,
        "batch_size": args.batch_size,
        "prediction_range": args.prediction_range,
        "trajectory": args.trajectory,
        "shuffle": True,
        "drop_last": True,
        "num_workers": 0,
        "cache_dir": args.cache_dir,
        "use_modal_cache": args.use_modal_cache,
        "seed": args.seed,
    }
    train_loader, num_samples = create_dataloader(**train_loader_args)
    print(f"Created data loader with {num_samples} training samples")
    if args.valid_start and args.valid_end and args.valid_start <= args.end_time:
        raise ValueError(f"Validation time interval has to start later than the training interval ends, got {args.valid_start} !< {args.end_time}")

    # Create valid loader
    print(f"Creating valid loader with with time range from {args.valid_start} to {args.valid_end}")
    valid_loader, num_samples = create_dataloader(
        data_path=args.data_path,
        start_time=args.valid_start,
        end_time=args.valid_end,
        coords=coords,
        physics_specs=physics_specs,
        batch_size=args.batch_size,
        prediction_range=args.prediction_range,
        trajectory=args.trajectory,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        cache_dir=args.cache_dir,
        use_modal_cache=args.use_modal_cache,
        seed=args.seed,
    )
    print(f"Created valid loader with {num_samples} training samples")

    
    
    # Initialize WandB
    if not args.no_wandb:
        runtype = 'train' if not args.test else 'test'
        wandb.init(
            name=f"{runtype}-{args.resolution}-{args.correction}-{args.prediction_range}-{args.train_length}",
            project=args.wandb_project,
            dir=os.environ.get("WANDB_DIR", ".")
        )
        wandb.config.update(args)
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("valid/*", step_metric="valid_step")

    # Build separate restore/save checkpoint managers
    restore_checkpoint_manager = None
    save_checkpoint_manager = None

    # Derive default subdir names when not provided
    save_subdir = args.save_checkpoint_name
    load_subdir = args.load_checkpoint_name
    if args.checkpoint_name is not None:
        if save_subdir is None:
            save_subdir = f"pr{args.prediction_range}"
        if load_subdir is None:
            # by default, load from the same subdir we save to
            load_subdir = save_subdir

    if args.checkpoint_name is not None:
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1, create=True)
        orbax_checkpointer = orbax.checkpoint.AsyncCheckpointer(
            orbax.checkpoint.PyTreeCheckpointHandler(),
            timeout_secs=50,
        )
        # Save manager always created if we have a base name
        save_checkpoint_path = CHKPT_DIR + f"/{args.checkpoint_name}"
        if save_subdir is not None:
            save_checkpoint_path += f"/{save_subdir}"
        save_checkpoint_manager = orbax.checkpoint.CheckpointManager(
            save_checkpoint_path, orbax_checkpointer, options
        )

        if args.load_checkpoint:
            restore_checkpoint_path = CHKPT_DIR + f"/{args.checkpoint_name}"
            if load_subdir is not None:
                restore_checkpoint_path += f"/{load_subdir}"
            restore_checkpoint_manager = orbax.checkpoint.CheckpointManager(
                restore_checkpoint_path, orbax_checkpointer, options
            )
    # Train model
    if not args.test:
        # Compute dycore parameters and instantiate dynamic core
        total_time = args.prediction_range * 6 * 60 * 60
        num_dycore_steps = total_time // int(args.dt_model)
        num_corrections = total_time // args.correction_interval
        dycore_steps_per_correction = num_dycore_steps // num_corrections
        dycore = DynamicalCore(
            coords=coords,
            dt=physics_specs.nondimensionalize(args.dt_model * scales.units.second),
            physics_specs=physics_specs,
            num_corrections=num_corrections,
            dycore_steps_per_correction=dycore_steps_per_correction,
            cache_dir=args.cache_dir,
        )
        # Create hybrid model
        print(f"Creating hybrid model")
        model = HybridModel(
            coords=coords,
            dt_physics=physics_specs.nondimensionalize(
                args.dt_model * scales.units.second
            ),
            dt_model=args.dt_model,
            physics_specs=physics_specs,
            prediction_range=args.prediction_range,
            correction_interval=args.correction_interval,
            correction=args.correction,
            trajectory=args.trajectory,
            gnn_hidden_dims=tuple(args.gnn_hidden_dims),
            cache_dir=args.cache_dir,
            dycore=dycore,
        )

        output = init_and_train_model(
            model=model,
            train_loader=train_loader,
            train_loader_args=train_loader_args,
            valid_loader=valid_loader,
            train_length=args.train_length,
            valid_interval=args.valid_interval,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate * jax.device_count(),
            log_wandb=not args.no_wandb,
            load_checkpoint=args.load_checkpoint,
            restore_checkpoint_manager=restore_checkpoint_manager,
            save_checkpoint_manager=save_checkpoint_manager,
            correction=args.correction,
            seed=args.seed,
        )
    elif args.test:
        output = None
        if args.test_year:
            test_start_time = get_start_time(args.test_year, args.prediction_range)
            test_end_time = get_end_time(args.test_year, args.prediction_range, args.batch_size)
            test_year = args.test_year
        else:
            test_start_time = datetime.strptime(args.start_time, "%Y-%m-%d") - timedelta(hours=args.prediction_range * 6)
            test_end_time = datetime.strptime(args.end_time, "%Y-%m-%d") + timedelta(hours=args.prediction_range * 6)
            test_start_time = test_start_time.strftime("%Y-%m-%d %H:%M:%S")
            test_end_time = test_end_time.strftime("%Y-%m-%d %H:%M:%S")

            print(f"Test start time: {test_start_time}, test end time: {test_end_time}")

            test_year = args.start_time.split('-')[0]

        total_time = args.prediction_range * 6 * 60 * 60
        num_dycore_steps = total_time // int(args.dt_model)
        num_corrections = total_time // args.correction_interval
        dycore_steps_per_correction = num_dycore_steps // num_corrections

        dycore = DynamicalCore(
            coords=coords,
            dt=physics_specs.nondimensionalize(args.dt_model * scales.units.second),
            physics_specs=physics_specs,
            num_corrections=num_corrections,
            dycore_steps_per_correction=dycore_steps_per_correction,
            cache_dir=args.cache_dir,
        )
        
        # Create hybrid model
        print(f"Creating hybrid model")
        model = HybridModel(
        coords=coords,
        dt_physics=physics_specs.nondimensionalize(
            args.dt_model * scales.units.second
        ),
        dt_model=args.dt_model,
        physics_specs=physics_specs,
        prediction_range=args.prediction_range,
        correction_interval=args.correction_interval,
        correction=args.correction,
        trajectory=args.trajectory,
        gnn_hidden_dims=tuple(args.gnn_hidden_dims),
        cache_dir=args.cache_dir,
        dycore=dycore,
        save_trajectories=args.save_trajectories,
        save_gain=args.save_gain,
        )

        test_loader, num_samples, time_values = create_dataloader(
            data_path=args.data_path,
            start_time=test_start_time,
            end_time=test_end_time,
            coords=coords,
            physics_specs=physics_specs,
            batch_size=args.batch_size,
            prediction_range=args.prediction_range,
            trajectory=args.trajectory,
            shuffle=False,
            drop_last=True,
            num_workers=0,
            cache_dir=args.cache_dir,
            use_modal_cache=args.use_modal_cache,
            seed=args.seed,
            xarray=args.xarray,
        )
        test_model(
            model=model,
            xarray=args.xarray,
            results_path=args.results_path,
            results_id=args.results_id,
            prediction_range=args.prediction_range,
            test_year=test_year,
            test_loader=test_loader,
            checkpoint_manager=restore_checkpoint_manager,
            time_values=time_values,
            learning_rate=args.learning_rate,
            save_trajectories=args.save_trajectories,
            save_gain=args.save_gain,
        )
    
    wandb.finish()
    return output


if __name__ == "__main__":
    main()
