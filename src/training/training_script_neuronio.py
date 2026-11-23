import gc
import json
import math
import os
import random
import time
from pathlib import Path

# configure JAX GPU memory preallocation before importing JAX
_parallel_executions = int(os.environ.get("CSTM_PRLL_EXEC", "1"))
os.environ.setdefault(
    "XLA_PYTHON_CLIENT_MEM_FRACTION", str(0.95 * (1 / _parallel_executions))
)

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.datasets.neuronio.neuronio_data_loader import NeuronIO
from src.datasets.neuronio.neuronio_data_utils import (
    DEFAULT_Y_SOMA_THRESHOLD,
    NEURONIO_DATA_DIM,
    NEURONIO_LABEL_DIM,
    create_neuronio_input_type,
    get_data_files_from_folder,
    parse_sim_experiment_file,
    visualize_training_batch,
)
from src.datasets.neuronio.neuronio_eval_utils import (
    NeuronioEvaluator,
    compute_test_predictions_multiple_sim_files,
    filter_and_extract_core_results,
)
from src.datasets.neuronio.neuronio_train_utils import neuronio_make_step
from src.datasets.neuronio.neuronio_viz_utils import visualize_neuron_workings
from src.models.elm_neuron import ELM
from src.training.train_utils import (
    calculate_model_cost_and_weight_stats,
    copy_model_to_cpu,
)


@hydra.main(
    version_base="1.3",
    config_path=".",
    config_name="curr_experiment_config",
)
def main(cfg: DictConfig):
    print("---------- Experiment started! ----------")

    # random delay for async start
    if not cfg.setup.debug_run:
        sleep_time = random.randint(1, 300)
        print(f"Sleeping: {sleep_time}s")
        time.sleep(sleep_time)
    else:
        print(f"Sleeping: 0s")

    ########## Logging ##########
    print("---------- Logging setup started: ----------")

    # artifacts directory
    experiment_base_dir = Path(os.getcwd())
    artifacts_dir = experiment_base_dir / "artifacts"
    os.makedirs(artifacts_dir)

    # get hydra config
    main_config = OmegaConf.to_container(cfg, resolve=True)
    with open(str(artifacts_dir / "main_config.json"), "w") as f:
        json.dump(main_config, f, indent=4)

    # wandb related setup
    if cfg.setup.wandb_logging:

        # login to wandb
        os.environ["WANDB_INIT_TIMEOUT"] = "300"
        entity = str(cfg.setup.wandb_entity_name)
        project_name = str(cfg.setup.wandb_project_name)
        with open(str(cfg.setup.wandb_api_key_path), "r") as file:
            api_key = file.read().strip()
        wandb.login(key=api_key)

        # create new run
        wandb.init(
            entity=entity,
            project=project_name,
            group=cfg.setup.wandb_group_name,
            config=main_config,
            save_code=False,
        )
        time.sleep(10)

    print("Experiment Base Directory:")
    print(experiment_base_dir)

    # Save the config properly
    print("Experiment Configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    ########## Seeding ##########
    print("---------- Seeding started: ----------")

    curr_actual_seed = cfg.setup.seed
    print(f"Current Actual Seed: {curr_actual_seed}")

    # report memory prealloc jax
    print(f"GPU Memory Fraction: {os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']}")

    # check backend working
    jax.config.update("jax_default_matmul_precision", cfg.setup.matmul_precision)
    print(f"Jax matmul precision {cfg.setup.matmul_precision}.")
    print("Jax CUDA available: ", jax.default_backend() == "gpu")
    print("Jaxlib binding works: ", bool(jax.numpy.ones(()) == 1))
    print("Jax Num Devices: ", len(jax.devices()))

    # general seeding
    random.seed(curr_actual_seed)
    np.random.seed(curr_actual_seed)

    # train seeding
    key = jrandom.PRNGKey(curr_actual_seed)
    key, init_key, train_key = jrandom.split(key, 3)

    # evaluation seeding
    const_eval_seed = 12345
    eval_key = jrandom.PRNGKey(const_eval_seed)
    eval_key, stats_key, viz_key, valid_key, test_key = jrandom.split(eval_key, 5)

    ########## Data ##########
    print("---------- Data configuration started: ----------")

    # data location
    datasets_path = Path(cfg.setup.datasets_base_folder)
    data_dir_path = datasets_path / "neuronio"
    train_data_dir_path = data_dir_path / "train"
    test_data_dir_path = data_dir_path / "test" / "Data_test"

    # sata splitting hardcoded
    train_data_dirs = [
        str(train_data_dir_path / "full_ergodic_train_batch_2"),
        str(train_data_dir_path / "full_ergodic_train_batch_3"),
        str(train_data_dir_path / "full_ergodic_train_batch_4"),
        str(train_data_dir_path / "full_ergodic_train_batch_5"),
        str(train_data_dir_path / "full_ergodic_train_batch_6"),
        str(train_data_dir_path / "full_ergodic_train_batch_7"),
        str(train_data_dir_path / "full_ergodic_train_batch_8"),
        str(train_data_dir_path / "full_ergodic_train_batch_9"),
        str(train_data_dir_path / "full_ergodic_train_batch_10"),
    ]
    valid_data_dirs = [str(train_data_dir_path / "full_ergodic_train_batch_1")]
    test_data_dirs = [str(test_data_dir_path)]

    train_files = get_data_files_from_folder(train_data_dirs)
    valid_files = get_data_files_from_folder(valid_data_dirs)
    test_files = get_data_files_from_folder(test_data_dirs)

    # preparing data oaders
    train_data_loader = NeuronIO(
        file_paths=train_files,
        batches_per_epoch=cfg.training.batches_per_epoch,
        batch_size=cfg.training.batch_size,
        input_window_size=cfg.training.input_window_size,
        file_load_fraction=cfg.training.file_load_fraction,
        num_workers=cfg.training.num_workers,
        num_prefetch_batch=cfg.training.num_prefetch_batch,
        seed=curr_actual_seed,
    )

    # variable valid file selection
    valid_key, train_file_select, valid_file_select = jrandom.split(valid_key, 3)

    # training data evaulation files
    train_file_indice = jrandom.randint(
        train_file_select, shape=(1,), minval=0, maxval=len(train_files)
    ).item()
    train_evaluator = NeuronioEvaluator(
        test_file=train_files[train_file_indice],
        input_window_size=cfg.training.input_window_size,
        burn_in_time=cfg.training.burn_in_time,
    )

    # validation data evaluation files
    valid_file_indice = jrandom.randint(
        valid_file_select, shape=(1,), minval=0, maxval=len(valid_files)
    ).item()
    valid_evaluator = NeuronioEvaluator(
        test_file=valid_files[valid_file_indice],
        input_window_size=cfg.training.input_window_size,
        burn_in_time=cfg.training.burn_in_time,
    )

    # example train batch
    X_viz, (y_spike_viz, y_soma_viz) = next(iter(train_data_loader))
    X_viz = np.asarray(X_viz)
    y_spike_viz = np.asarray(y_spike_viz)
    y_soma_viz = np.asarray(y_soma_viz)

    # apples to apples comparison
    sample_file_name = "100523.p"
    sample_sequence_index = 33
    sample_select_start = 2000
    sample_select_len = 2000

    # load file
    select_test_file = [file for file in test_files if sample_file_name in file][0]
    viz_X, viz_y_spike, viz_y_soma = parse_sim_experiment_file(select_test_file)

    # select sequence
    viz_X = viz_X[
        :,
        sample_select_start : sample_select_start + sample_select_len,
        sample_sequence_index,
    ]
    viz_y_spike = viz_y_spike[
        sample_select_start : sample_select_start + sample_select_len,
        sample_sequence_index,
    ]
    viz_y_soma = viz_y_soma[
        sample_select_start : sample_select_start + sample_select_len,
        sample_sequence_index,
    ]

    # preprocess sequence
    synapse_types = create_neuronio_input_type()
    viz_X = viz_X.T * synapse_types[:]
    viz_y_soma = np.clip(viz_y_soma, max=DEFAULT_Y_SOMA_THRESHOLD)

    ########## Model ##########
    print("---------- Model configuration started: ----------")

    # get the model config
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config["num_input"] = NEURONIO_DATA_DIM
    model_config["num_output"] = NEURONIO_LABEL_DIM

    # instantiate ELM neuron
    model = ELM(
        **model_config,
        key=init_key,
    )

    # serialize network config
    with open(str(artifacts_dir / "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    # trainable parameter specification
    train_params_filter = model.get_train_params_filter()

    # model visualization
    print("Model Architecture:")
    summary_str = str(model)
    print(summary_str)
    with open(str(artifacts_dir / "model_summary.txt"), "w") as f:
        f.write(summary_str)

    # model statistics
    print("Model Stats:")
    model_stats = calculate_model_cost_and_weight_stats(
        model,
        train_params_filter,
        data_shape=(1, NEURONIO_DATA_DIM),
        comp_key=stats_key,
    )
    print(json.dumps(model_stats, indent=4))
    with open(str(artifacts_dir / "model_stats.json"), "w") as f:
        json.dump(model_stats, f, indent=4)
    model_cost = {
        "bytes accessed": model_stats["cost_analysis"].get("bytes accessed", 0.0),
        "flops": model_stats["cost_analysis"].get("flops", 0.0),
        "transcendentals": model_stats["cost_analysis"].get("transcendentals", 0.0),
        "total_params": model_stats["total_params"],
        "total_trainable_params": model_stats["total_trainable_params"],
    }
    if cfg.setup.wandb_logging:
        wandb.log({"model_cost": model_cost})

    ########## Data & Pred Visualization ##########
    print("---------- Initial visualizaiton started: ----------")

    # deterministic evaluation
    viz_key, batch_viz_key, inner_working_viz_key = jrandom.split(viz_key, 3)

    # visualize predictions batch
    X_viz_jax = jnp.asarray(X_viz, dtype=jnp.float32)
    batch_viz_key_keys = jrandom.split(batch_viz_key, X_viz.shape[0])
    outputs, _, _ = jax.vmap(lambda x, key: model.neuronio_eval_forward(x=x, key=key))(
        X_viz_jax, batch_viz_key_keys
    )
    visualize_training_batch(
        input_spikes=X_viz,
        target_spikes=y_spike_viz,
        target_soma=y_soma_viz,
        pred_spikes=outputs[..., 0],
        pred_soma=outputs[..., 1],
        num_viz=8,
        save_fig_path=str(artifacts_dir / "example_batch"),
    )

    # visualize predictions sample
    viz_X_jax = jnp.asarray(viz_X, dtype=jnp.float32)
    outputs, recordings, _ = model.neuronio_eval_forward(
        x=viz_X_jax, monitor=["branch", "memory", "mlp"], key=inner_working_viz_key
    )
    visualize_neuron_workings(
        neuron_taus=model.tau_m,
        neuron_output=outputs,
        neuron_branch=recordings["branch"],
        neuron_memory=recordings["memory"],
        input_spikes=viz_X,
        target_spikes=viz_y_spike,
        target_soma=viz_y_soma,
        save_fig_path=str(artifacts_dir / "inner_workings"),
    )
    del outputs, recordings

    ########## Scheduler and Optimizer##########
    print("---------- Scheduler and Optimizer configuration started: ----------")

    # sheduler
    if cfg.training.scheduler == "cosine":
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.training.learning_rate,
            warmup_steps=cfg.training.warmup_steps,
            decay_steps=cfg.training.batches_per_epoch * cfg.training.num_epochs,
            end_value=0.0,
        )
    else:
        raise ValueError("Unsupported scheduler")

    # optimizer
    if cfg.training.optimizer == "adam":
        optimizer = optax.adam(learning_rate=scheduler)
    elif cfg.training.optimizer == "adamax":
        optimizer = optax.adamax(learning_rate=scheduler)
    else:
        raise ValueError("Unsupported optimizer")

    # gradient clipping
    grad_clip = optax.clip_by_global_norm(cfg.training.clip_norm)

    gradient_transform = optax.chain(
        grad_clip,
        optimizer,
    )

    opt_state = gradient_transform.init(eqx.filter(model, train_params_filter))

    ########## Training START ##########
    print("---------- Training started: ----------")

    best_model_file_name = "best_model_state_dict.eqx"
    best_model_state_dict = copy_model_to_cpu(model)
    eqx.tree_serialise_leaves(
        str(artifacts_dir / best_model_file_name), best_model_state_dict
    )

    # set global variable
    train_step = 0
    train_total_loss = 0.0
    train_loss = float("nan")
    valid_metric = float("inf") if cfg.training.select_rmse else 0.0
    train_start_time = time.time()
    best_model_epoch = 0
    valid_every_epochs = max(1, int(cfg.training.get("valid_every_epochs", 1)))

    # deterministic validation
    valid_key, train_valid_eval_key, valid_valid_eval_key = jrandom.split(valid_key, 3)

    for train_epoch in range(1, cfg.training.num_epochs + 1):
        # ------- START OF EPOCH -------
        epoch_loss_sum = jnp.zeros(())
        pbar = tqdm(
            enumerate(train_data_loader, 0),
            initial=train_step,
            total=cfg.training.batches_per_epoch * cfg.training.num_epochs,
            disable=not cfg.setup.debug_run,
        )
        for _, data in pbar:
            # ------- START OF STEP -------

            inputs, targets = data
            train_key, make_step_key = jrandom.split(train_key, 2)

            # Perform a single training step
            loss, _, model, opt_state, _ = neuronio_make_step(
                model,
                inputs,
                targets,
                opt_state,
                make_step_key,
                batch_size=cfg.training.batch_size,
                train_params_filter=train_params_filter,
                gradient_transform=gradient_transform,
                burn_in_time=cfg.training.burn_in_time,
                init_carry=None,
                return_grad_norms=False,
                return_carry=False,
            )
            train_step += 1

            # accumulate loss on device (avoids per-step host sync)
            epoch_loss_sum = epoch_loss_sum + loss

            # ------- END OF STEP -------

        # flush epoch loss from device once per epoch
        epoch_loss = float(epoch_loss_sum)
        if not math.isfinite(epoch_loss):
            print("Loss is nan: exiting training.")
            break
        train_total_loss += epoch_loss
        train_loss = train_total_loss / max(1, train_step)

        # ------- VALIDAITON -------

        validate = (
            train_epoch % valid_every_epochs == 0
            or train_epoch == cfg.training.num_epochs
        )
        if validate:
            # evaluate on training data
            train_eval_metrics = train_evaluator.evaluate(model, train_valid_eval_key)
            train_rmse = train_eval_metrics["soma_RMSE"]
            train_auc = train_eval_metrics["AUC"]

            # evaluate on validation data
            valid_eval_metrics = valid_evaluator.evaluate(model, valid_valid_eval_key)
            valid_rmse = valid_eval_metrics["soma_RMSE"]
            valid_auc = valid_eval_metrics["AUC"]

            # check whether model has improved
            if cfg.training.select_rmse:
                if valid_rmse < valid_metric:
                    best_model_epoch = train_epoch
                    valid_metric = valid_rmse
                    best_model_state_dict = copy_model_to_cpu(model)
                    eqx.tree_serialise_leaves(
                        str(artifacts_dir / best_model_file_name),
                        best_model_state_dict,
                    )
            else:
                if valid_auc > valid_metric:
                    best_model_epoch = train_epoch
                    valid_metric = valid_auc
                    best_model_state_dict = copy_model_to_cpu(model)
                    eqx.tree_serialise_leaves(
                        str(artifacts_dir / best_model_file_name),
                        best_model_state_dict,
                    )

        # logging of epoch metrics
        train_time = time.time() - train_start_time
        train_log = {
            "train_epoch": int(train_epoch),
            "train_time": float(train_time),
            "train_loss": float(train_loss),
        }
        if validate:
            train_log.update(
                {
                    "train_rmse": float(train_rmse),
                    "train_auc": float(train_auc),
                    "valid_rmse": float(valid_rmse),
                    "valid_auc": float(valid_auc),
                }
            )
        train_log.update(
            {
                "learning_rate": float(scheduler(train_step)),
                "best_model_epoch": int(best_model_epoch),
            }
        )
        print(json.dumps(train_log, separators=(",", ":")))
        if cfg.setup.wandb_logging:
            wandb.log({"train_log": train_log})

        # ------- END OF EPOCH -------

    # free up memory
    del train_data_loader
    gc.collect()

    ########## Evaluaiton ##########
    print("---------- Evaluation started: ----------")

    # load the best model
    print(f"Loading best model from epoch: {best_model_epoch}...")
    model = eqx.tree_deserialise_leaves(
        str(artifacts_dir / best_model_file_name), model
    )

    # seeding & determinism
    random.seed(const_eval_seed)
    np.random.seed(const_eval_seed)
    eval_key = jrandom.PRNGKey(const_eval_seed)
    eval_key, stats_key, viz_key, valid_key, test_key = jrandom.split(eval_key, 5)

    # PRIMARY QUALITATIVE EVALUSTION

    # deterministic evaluation
    viz_key, batch_viz_key, inner_working_viz_key = jrandom.split(viz_key, 3)

    # visualize predictions batch
    X_viz_jax = jnp.asarray(X_viz, dtype=jnp.float32)
    batch_viz_key_keys = jrandom.split(batch_viz_key, X_viz.shape[0])
    outputs, _, _ = jax.vmap(lambda x, key: model.neuronio_eval_forward(x=x, key=key))(
        X_viz_jax, batch_viz_key_keys
    )
    visualize_training_batch(
        input_spikes=X_viz,
        target_spikes=y_spike_viz,
        target_soma=y_soma_viz,
        pred_spikes=outputs[..., 0],
        pred_soma=outputs[..., 1],
        num_viz=8,
        save_fig_path=str(artifacts_dir / "example_batch_after"),
    )

    # visualize predictions sample
    viz_X_jax = jnp.asarray(viz_X, dtype=jnp.float32)
    outputs, recordings, _ = model.neuronio_eval_forward(
        x=viz_X_jax, monitor=["branch", "memory", "mlp"], key=inner_working_viz_key
    )
    visualize_neuron_workings(
        neuron_taus=model.tau_m,
        neuron_output=outputs,
        neuron_branch=recordings["branch"],
        neuron_memory=recordings["memory"],
        input_spikes=viz_X,
        target_spikes=viz_y_spike,
        target_soma=viz_y_soma,
        save_fig_path=str(artifacts_dir / "inner_workings_after"),
    )

    # PRIMARY QUANTITIVVE EVALUSTION

    # deterministic evaluation
    test_key, train_file_test_select, train_test_key, valid_test_key, test_test_key = (
        jrandom.split(test_key, 5)
    )

    train_file_indices = jrandom.randint(
        train_file_test_select, shape=(10,), minval=0, maxval=len(train_files)
    ).tolist()
    select_train_files = [train_files[idx] for idx in train_file_indices]
    train_predictions = compute_test_predictions_multiple_sim_files(
        neuron=model,
        test_files=(
            select_train_files if not cfg.setup.debug_run else select_train_files[:1]
        ),
        input_window_size=cfg.training.input_window_size,
        burn_in_time=cfg.training.burn_in_time,
        key=train_test_key,
    )
    train_results = filter_and_extract_core_results(*train_predictions)

    valid_predictions = compute_test_predictions_multiple_sim_files(
        neuron=model,
        test_files=valid_files if not cfg.setup.debug_run else valid_files[:1],
        input_window_size=cfg.training.input_window_size,
        burn_in_time=cfg.training.burn_in_time,
        key=valid_test_key,
    )
    valid_results = filter_and_extract_core_results(*valid_predictions)

    test_predictions = compute_test_predictions_multiple_sim_files(
        neuron=model,
        test_files=test_files if not cfg.setup.debug_run else test_files[:1],
        input_window_size=cfg.training.input_window_size,
        burn_in_time=cfg.training.burn_in_time,
        key=test_test_key,
    )
    test_results = filter_and_extract_core_results(*test_predictions)

    eval_results = dict()
    eval_results["train_results"] = train_results
    eval_results["valid_results"] = valid_results
    eval_results["test_results"] = test_results

    print(json.dumps(eval_results, indent=4))
    with open(str(artifacts_dir / "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)
    if cfg.setup.wandb_logging:
        wandb.log({"eval_results": eval_results})

    ########## Cleanup ##########
    print("---------- Cleanup started: ----------")

    # save artefacts to wandb
    if cfg.setup.wandb_logging:
        files_to_upload = os.listdir(str(artifacts_dir))
        if not cfg.setup.wandb_upload_weights:
            files_to_upload.remove(best_model_file_name)
        for file_name in files_to_upload:
            file_path = str(artifacts_dir / file_name)
            wandb.save(file_path, base_path=str(experiment_base_dir), policy="now")
        wandb.finish()

    ########## FIN ##########
    print("---------- Experiment finished! ----------")


if __name__ == "__main__":
    main()
