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
import h5py
import hydra
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.sharding as jshard
import numpy as np
import optax
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.datasets.shd.shd_data_loader import SHD, SHDAdding, random_val_split_SHD_data
from src.datasets.shd.shd_train_utils import shd_loss, shd_make_step
from src.datasets.shd.shd_viz_utils import visualize_network_inference
from src.models.elm_network import ELMNetwork
from src.training.regularizer import ELMNetworkRegularizer
from src.training.train_utils import (
    calculate_model_cost_and_weight_stats,
    cast_floats,
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
    print("---------- Setup started: ----------")

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

    # get the downloaded dataset
    dataset_path = Path(cfg.setup.datasets_base_folder) / "heidelberg"
    train_file = h5py.File(os.path.join(str(dataset_path), "shd_train.h5"), "r")
    test_file = h5py.File(os.path.join(str(dataset_path), "shd_test.h5"), "r")

    # training and validation splitting
    x_train, y_train, x_valid, y_valid = random_val_split_SHD_data(
        train_file=train_file,
        valid_fraction=cfg.training.valid_fraction,
        seed=int(const_eval_seed),
    )

    # extract the testing dataset
    x_test = test_file["spikes"]
    y_test = test_file["labels"]

    # initialize dataloaders
    valid_batch_size = int(
        cfg.training.get("valid_batch_size", cfg.training.batch_size)
    )
    if not cfg.training.digit_addition:
        train_dataset = SHD(
            X=x_train,
            y=y_train,
            batch_size=cfg.training.batch_size,
            bin_size=cfg.training.bin_size,
            shuffle=True,
            test_set=False,
            seed=curr_actual_seed,
        )
        valid_dataset = SHD(
            X=x_valid,
            y=y_valid,
            batch_size=valid_batch_size,
            bin_size=cfg.training.bin_size,
            shuffle=False,
            test_set=False,
        )
        test_dataset = SHD(
            X=x_test,
            y=y_test,
            batch_size=cfg.training.eval_batch_size,
            bin_size=cfg.training.bin_size,
            shuffle=False,
        )

        batches_per_epoch = len(train_dataset) if not cfg.setup.debug_run else 20
        valid_batches_per_epoch = len(valid_dataset) if not cfg.setup.debug_run else 20
        test_batches_per_epoch = len(test_dataset) if not cfg.setup.debug_run else 20
    else:
        batches_per_epoch = 2000 if not cfg.setup.debug_run else 20
        valid_batches_per_epoch = 500 if not cfg.setup.debug_run else 20
        test_batches_per_epoch = 2000 if not cfg.setup.debug_run else 20

        train_dataset = SHDAdding(
            X=x_train,
            y=y_train,
            batch_size=cfg.training.batch_size,
            bin_size=cfg.training.bin_size,
            batches_per_epoch=batches_per_epoch,
            shuffle=True,
            seed=curr_actual_seed,
        )
        valid_dataset = SHDAdding(
            X=x_valid,
            y=y_valid,
            batch_size=valid_batch_size,
            bin_size=cfg.training.bin_size,
            batches_per_epoch=valid_batches_per_epoch,
            shuffle=False,
        )
        test_dataset = SHDAdding(
            X=x_test,
            y=y_test,
            batch_size=cfg.training.eval_batch_size,
            bin_size=cfg.training.bin_size,
            batches_per_epoch=test_batches_per_epoch,
            shuffle=False,
        )

    data_num_input_channel = 700
    data_num_classes = 20 if not cfg.training.digit_addition else 19
    data_num_time_bins = test_dataset.num_time_bins
    example_viz_batch = next(iter(test_dataset))

    ########## Model ##########
    print("---------- Model configuration started: ----------")

    # get the model config
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config["num_input"] = data_num_input_channel
    model_config["num_output"] = data_num_classes

    # model initialization
    model = ELMNetwork(
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
        data_shape=(1, data_num_input_channel),
        comp_key=stats_key,
        data_dtype=jnp.bool,
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
    print("---------- Initial visualization started: ----------")

    # deterministic evaluation
    viz_key, batch_viz_key = jrandom.split(viz_key, 2)

    # convert data
    inputs = jnp.array(example_viz_batch[0], dtype=jnp.bool)
    labels = jnp.array(example_viz_batch[1], dtype=jnp.int32)

    # get predictions and recordings (only layer activity is needed for viz)
    viz_monitor = (["activity"], [])
    batch_viz_keys = jrandom.split(batch_viz_key, cfg.training.eval_batch_size)
    logits, recordings, _ = jax.vmap(
        lambda x, key: model(
            x=x,
            init_carry=None,
            key=key,
            monitor=viz_monitor,
            inference=True,
        )
    )(inputs, batch_viz_keys)
    recordings = cast_floats(recordings, "float32")

    # visualize predictions
    visualize_network_inference(
        inputs=inputs,
        labels=labels,
        logits=logits,
        recordings=recordings,
        bin_size=cfg.training.bin_size,
        adding=cfg.training.digit_addition,
        max_viz_samples=8,
        path=str(artifacts_dir / "data_and_pred_viz"),
    )
    del logits, recordings

    ########## Scheduler and Optimizer##########
    print("---------- Scheduler and Optimizer configuration started: ----------")

    # instantiate scheduler
    if cfg.training.scheduler == "cosine":
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.training.learning_rate,
            warmup_steps=cfg.training.warmup_steps if not cfg.setup.debug_run else 5,
            decay_steps=batches_per_epoch * cfg.training.num_epochs,
            end_value=0.0,
        )
    else:
        raise ValueError("Unsupported scheduler")

    # instantiate optimizer
    if cfg.training.optimizer == "adam":
        optimizer = optax.adam(learning_rate=scheduler)
    elif cfg.training.optimizer == "adamax":
        optimizer = optax.adamax(learning_rate=scheduler)
    else:
        raise ValueError("Unsupported optimizer")

    # apply gradient clipping
    grad_clip = optax.clip_by_global_norm(cfg.training.clip_norm)
    gradient_transform = optax.chain(
        grad_clip,
        optimizer,
    )

    # optimizer state
    opt_state = gradient_transform.init(eqx.filter(model, train_params_filter))

    # setup regularizer
    reg_config = OmegaConf.select(cfg, "training.reg_config", default=None)
    regularizer = (
        None
        if reg_config is None
        else ELMNetworkRegularizer(OmegaConf.to_container(reg_config, resolve=True))
    )

    ########## MULTI GPU SHARING ##########
    print("---------- Multi GPU sharding initialization: ----------")

    num_devices = len(jax.devices())
    mesh_kwargs = (
        {"axis_types": (jshard.AxisType.Auto,)} if hasattr(jshard, "AxisType") else {}
    )
    mesh = jax.make_mesh((num_devices,), ("batch",), **mesh_kwargs)
    data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("batch"))
    model_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec())

    # sharding model and opt state
    model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)

    ########## EVAL FUNCTION ##########
    print("---------- Eval function defined: ----------")

    def evaluate_model(
        model,
        eval_key,
        eval_iter,
        batch_size: int,
        num_classes: int,
        model_data_sharding=None,
        debug_log: bool = False,
    ):
        total_predictions = 0
        gpu_loss = jnp.zeros(())
        gpu_correct = jnp.zeros((), jnp.int32)
        if model_data_sharding is not None:
            model = eqx.filter_shard(model, model_data_sharding[0])
        for _, data in tqdm(
            enumerate(eval_iter, 0), total=len(eval_iter), disable=not debug_log
        ):
            inputs, labels = data
            eval_key, inference_key = jrandom.split(eval_key, 2)

            # Conversion to JNP
            inputs = jnp.array(inputs, dtype=jnp.bool)
            labels = jnp.array(labels, dtype=jnp.int32)

            # sharding data
            if model_data_sharding is not None:
                inputs, labels = eqx.filter_shard(
                    (inputs, labels), model_data_sharding[1]
                )

            # Only compute loss
            loss, (logits, _) = shd_loss(
                model,
                inputs,
                labels,
                inference_key,
                batch_size,
                num_classes,
                inference=True,
                init_carry=None,
                return_logits=True,
                return_carry=False,
                label_smoothing=0.0,  # eval
                regularizer=None,  # eval
            )
            gpu_loss = gpu_loss + loss

            # Calculate accuracy on device
            predicted = jnp.argmax(logits, 1)
            gpu_correct = gpu_correct + jnp.sum(predicted == labels, dtype=jnp.int32)
            total_predictions += labels.shape[0]

        total_loss, correct_predictions = jax.device_get((gpu_loss, gpu_correct))
        eval_loss = float(total_loss) / len(eval_iter)
        eval_accuracy = int(correct_predictions) / total_predictions

        return eval_loss, eval_accuracy

    ########## Training START ##########
    print("---------- Training started: ----------")

    best_model_file_name = "best_model_state_dict.eqx"
    best_model_state_dict = copy_model_to_cpu(model)
    eqx.tree_serialise_leaves(
        str(artifacts_dir / best_model_file_name), best_model_state_dict
    )

    # set global variable
    train_step = 0
    train_loss = float("nan")
    train_start_time = time.time()
    best_valid_accuracy = 0.0
    best_model_epoch = 0
    best_model_train_loss = float("nan")
    best_model_train_accuracy = 0.0
    valid_accuracy = 0.0

    valid_every_epochs = max(1, int(cfg.training.get("valid_every_epochs", 1)))
    grad_checkpointing = bool(cfg.training.get("grad_checkpointing", False))

    # termination by step, not by epoch
    for train_epoch in range(1, cfg.training.num_epochs + 1):
        # ------- START OF EPOCH -------

        # running statistics on device (avoids per-step host sync)
        epoch_loss_sum = jnp.zeros(())
        epoch_step_count = 0
        train_correct_predictions = jnp.zeros((), jnp.int32)
        train_total_predictions = 0

        # configure pbar
        pbar = tqdm(
            enumerate(train_dataset, 0),
            initial=train_step,
            total=batches_per_epoch * cfg.training.num_epochs,
            disable=not cfg.setup.debug_run,
        )
        for epoch_step, (inputs, labels) in pbar:
            if epoch_step >= batches_per_epoch:
                break
            # ------- START OF STEP -------

            # step tracking
            train_step += 1

            # convert data
            inputs = jnp.array(inputs, dtype=jnp.bool)
            labels = jnp.array(labels, dtype=jnp.int32)

            # sharding data
            inputs, labels = eqx.filter_shard((inputs, labels), data_sharding)

            # make step
            train_key, make_step_key = jrandom.split(train_key, 2)
            loss, logits, _, model, opt_state, _ = shd_make_step(
                model,
                inputs,
                labels,
                opt_state,
                make_step_key,
                batch_size=cfg.training.batch_size,
                num_classes=data_num_classes,
                train_params_filter=train_params_filter,
                gradient_transform=gradient_transform,
                init_carry=None,
                return_logits=True,
                return_grad_norms=False,
                return_carry=False,
                label_smoothing=cfg.training.label_smoothing,
                regularizer=regularizer,
                grad_checkpointing=grad_checkpointing,
            )

            # accumulate loss and accuracy on device (no host sync per step)
            epoch_loss_sum = epoch_loss_sum + loss
            epoch_step_count += 1
            predicted = jnp.argmax(logits, 1)
            train_correct_predictions = train_correct_predictions + jnp.sum(
                predicted == labels, dtype=jnp.int32
            )
            train_total_predictions += labels.shape[0]

            # ------- END OF STEP -------

        # flush epoch metrics from device once per epoch
        loss_sum, correct_sum = jax.device_get(
            (epoch_loss_sum, train_correct_predictions)
        )
        if not math.isfinite(float(loss_sum)):
            print("Loss is nan: exiting training.")
            break
        train_loss = float(loss_sum) / max(1, epoch_step_count)
        train_accuracy = int(correct_sum) / max(1, train_total_predictions)

        # ------- VALIDATION -------

        validate = (
            train_epoch % valid_every_epochs == 0
            or train_epoch == cfg.training.num_epochs
        )
        if validate:
            # evaluate model
            valid_loss, valid_accuracy = evaluate_model(
                model,
                valid_key,
                valid_dataset,
                valid_batch_size,
                data_num_classes,
                model_data_sharding=(model_sharding, data_sharding),
                debug_log=cfg.setup.debug_run,
            )

            # update best model
            if (
                not (valid_accuracy is None or math.isnan(valid_accuracy))
                and valid_accuracy > best_valid_accuracy
            ):
                best_valid_accuracy = valid_accuracy
                best_model_train_loss = train_loss
                best_model_train_accuracy = train_accuracy
                best_model_epoch = train_epoch
                best_model_state_dict = copy_model_to_cpu(model)
                eqx.tree_serialise_leaves(
                    str(artifacts_dir / best_model_file_name), best_model_state_dict
                )

        # logging
        train_time = time.time() - train_start_time
        train_log = {
            "train_epoch": int(train_epoch),
            "train_time": float(train_time),
            "train_loss": float(train_loss),
        }
        if validate:
            train_log["valid_loss"] = float(valid_loss)
        train_log["train_accuracy"] = float(train_accuracy)
        if validate:
            train_log["valid_accuracy"] = float(valid_accuracy)
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

    # ------- END OF TRAINING -------

    ########## Evaluation ##########
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

    ########## Qualitative Evaluation ##########
    print("---------- Qualitative evaluation started: ----------")

    # deterministic evaluation
    viz_key, batch_viz_key = jrandom.split(viz_key, 2)

    # convert data
    inputs = jnp.array(example_viz_batch[0], dtype=jnp.bool)
    labels = jnp.array(example_viz_batch[1], dtype=jnp.int32)

    # get predictions
    batch_viz_keys = jrandom.split(batch_viz_key, cfg.training.eval_batch_size)
    logits, recordings, _ = jax.vmap(
        lambda x, key: model(
            x=x,
            init_carry=None,
            key=key,
            monitor=viz_monitor,
            inference=True,
        )
    )(inputs, batch_viz_keys)
    recordings = cast_floats(recordings, "float32")

    # visualize predictions
    visualize_network_inference(
        inputs=inputs,
        labels=labels,
        logits=logits,
        recordings=recordings,
        bin_size=cfg.training.bin_size,
        adding=cfg.training.digit_addition,
        max_viz_samples=8,
        path=str(artifacts_dir / "data_and_pred_viz_after"),
    )
    del logits, recordings

    ########## Quantitative Evaluation ##########
    print("---------- Quantitative evaluation started: ----------")

    # deterministic evaluation
    test_key, valid_test_key, test_test_key = jrandom.split(test_key, 3)

    # Calculate valid performance
    valid_loss, valid_accuracy = evaluate_model(
        model,
        valid_test_key,
        valid_dataset,
        valid_batch_size,
        data_num_classes,
        model_data_sharding=(model_sharding, data_sharding),
        debug_log=cfg.setup.debug_run,
    )

    # Calculate test performance
    test_loss, test_accuracy = evaluate_model(
        model,
        test_test_key,
        test_dataset,
        cfg.training.eval_batch_size,  # fixed size 8 will be shardable
        data_num_classes,
        model_data_sharding=(model_sharding, data_sharding),
        debug_log=cfg.setup.debug_run,
    )

    # Logging evaluation metrics
    eval_results = {
        "final_train_loss": float(
            best_model_train_loss if best_model_epoch > 0 else train_loss
        ),
        "final_valid_loss": float(valid_loss),
        "final_test_loss": float(test_loss),
        f"final_train_accuracy": float(best_model_train_accuracy),
        f"final_valid_accuracy": float(valid_accuracy),
        f"final_test_accuracy": float(test_accuracy),
    }
    print(json.dumps(eval_results, indent=4))
    with open(str(artifacts_dir / "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)
    if cfg.setup.wandb_logging:
        wandb.log({"eval_results": eval_results})

    ########## Cleanup ##########
    print("---------- Cleanup started: ----------")

    # save artifacts to wandb
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
