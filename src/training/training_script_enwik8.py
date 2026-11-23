import gc
import json
import math
import os
import random
import time
import traceback
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
import jax.sharding as jshard
import numpy as np
import optax
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.datasets.language.data_loader import get_lm_corpus
from src.datasets.language.language_train_utils import language_loss, language_make_step
from src.datasets.language.language_viz_utils import (
    calculate_and_plot_spiking_stats,
    visualize_batch,
)
from src.models.elm_network import ELMNetwork
from src.training.regularizer import ELMNetworkRegularizer
from src.training.train_utils import (
    calculate_model_cost_and_weight_stats,
    cast_floats,
    copy_model_to_cpu,
    cosine_decay,
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

    # get data corpus
    dataset_path = Path(cfg.setup.datasets_base_folder) / "language_modeling"
    corpus = get_lm_corpus(str(dataset_path / "enwik8"), "enwik8")
    vocab_size = len(corpus.vocab)

    # get data iterators
    test_batch_size = int(cfg.training.get("test_batch_size", 1))
    viz_batch_size = min(cfg.training.valid_batch_size, 8)
    viz_seq_length = 2000
    train_iterator = corpus.get_iterator(
        "train",
        cfg.training.batch_size,
        cfg.training.seq_length,
        device="cpu",
        ext_len=0,
    )
    valid_iterator = corpus.get_iterator(
        "valid",
        cfg.training.valid_batch_size,
        cfg.training.seq_length,
        device="cpu",
        ext_len=0,
    )
    test_iterator = corpus.get_iterator(
        "test",
        test_batch_size,
        cfg.training.seq_length,
        device="cpu",
        ext_len=0,
    )
    viz_iterator = corpus.get_iterator(
        "test",
        viz_batch_size,
        viz_seq_length,
        device="cpu",
        ext_len=0,
    )

    # get test examples
    example_train_batch = train_iterator.get_batch(0)
    example_test_batch = test_iterator.get_batch(0)
    example_viz_batch = viz_iterator.get_batch(0)
    print("Train Batch shape: ", example_train_batch[0].shape)
    print("Test Batch shape: ", example_test_batch[0].shape)
    print("Viz Batch shape: ", example_viz_batch[0].shape)
    print("Vocab size: ", vocab_size)

    ########## Model ##########
    print("---------- Model configuration started: ----------")

    # get the model config
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config["num_input"] = vocab_size
    model_config["num_output"] = vocab_size

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
        data_shape=(1,),
        comp_key=stats_key,
        data_dtype=jnp.int32,
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
    inputs = jnp.array(example_viz_batch[0].T, dtype=jnp.int32)
    targets = jnp.array(example_viz_batch[1].T, dtype=jnp.int32)

    # get predictions and recordings (only activity and preact are needed for viz)
    viz_monitor = (["activity"], ["preact"])
    batch_viz_keys = jrandom.split(batch_viz_key, viz_batch_size)
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
    example_preds = logits.argmax(axis=-1).T
    viz_str = visualize_batch(
        data_corpus=corpus,
        example_batch=example_viz_batch,
        example_preds=example_preds,
        max_viz_samples=8,
        torch=False,
    )
    with open(artifacts_dir / "data_and_pred_viz.txt", "w") as f:
        f.write(viz_str)

    # visualize hidden layer activity
    batch_viz_idx, hidden_layer_idx = 0, 0
    last_num_steps = viz_seq_length // 2

    hidden_layer_activity = np.array(
        recordings[hidden_layer_idx][0]["activity"][batch_viz_idx, -last_num_steps:]
    )
    hidden_neuron_preacts = np.array(
        recordings[hidden_layer_idx][1]["preact"][batch_viz_idx, -last_num_steps:, :, 0]
    )
    del logits, recordings

    try:
        activity_stats = calculate_and_plot_spiking_stats(
            layer_activity=hidden_layer_activity,
            neuron_outputs=hidden_neuron_preacts,
            neuron_biases=model.layers[hidden_layer_idx].ensemble.b,
            max_interval=min(200, last_num_steps),
            path=str(artifacts_dir / "activity_hidden_and_stats"),
        )
        print("Hidden Layer Activity Stats:")
        print(json.dumps(activity_stats, indent=4))
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()

    ########## Scheduler and Optimizer##########
    print("---------- Scheduler and Optimizer configuration started: ----------")

    # instantiate scheduler
    if cfg.training.scheduler == "cosine":
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.training.learning_rate,
            warmup_steps=cfg.training.warmup_steps,
            decay_steps=cfg.training.steps_per_turn * cfg.training.num_turns,
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

    # configure regularizer
    reg_config = OmegaConf.select(cfg, "training.reg_config", default=None)
    regularizer = (
        None
        if reg_config is None
        else ELMNetworkRegularizer(OmegaConf.to_container(reg_config, resolve=True))
    )

    ########## MULTI GPU SHARING ##########
    print("---------- Multi GPU sharding initialization: ----------")

    # NOTE batch_size must be divisible by num_devices
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
        batch_size,
        vocab_size,
        train_eval_batches=None,
        hidden_state_reuse=False,
        model_data_sharding=None,
        debug_log=False,
    ):
        carry = None
        curr_eval_batch = 0
        train_eval_batches = (
            eval_iter.n_batch if train_eval_batches is None else train_eval_batches
        )
        # summed on device and read once at the end of eval
        total_len, gpu_loss = 0, jnp.zeros(())
        if model_data_sharding is not None:
            model = eqx.filter_shard(model, model_data_sharding[0])
        pbar_eval = tqdm(
            enumerate(eval_iter, 0),
            total=train_eval_batches,
            disable=not debug_log,
        )
        for _, (inputs, targets, curr_seq_len) in pbar_eval:
            # batch counting
            curr_eval_batch += 1

            # hidden state logic
            carry = carry if hidden_state_reuse else None

            # convert data
            inputs = jnp.array(inputs.T, dtype=jnp.int32)
            targets = jnp.array(targets.T, dtype=jnp.int32)

            # sharding data
            if model_data_sharding is not None:
                inputs, targets = eqx.filter_shard(
                    (inputs, targets), model_data_sharding[1]
                )

            # calculate loss
            eval_key, curr_eval_key = jrandom.split(eval_key, 2)
            loss, (_, carry) = language_loss(
                model,
                inputs,
                targets,
                curr_eval_key,
                inference=True,
                batch_size=batch_size,
                vocab_size=vocab_size,
                burn_in_time=0,  # eval
                label_smoothing=0.0,  # eval
                init_carry=carry,
                return_logits=False,
                return_carry=hidden_state_reuse,
                regularizer=None,  # eval
            )

            total_len += curr_seq_len
            gpu_loss = gpu_loss + loss * curr_seq_len

            # termination condition
            if curr_eval_batch >= train_eval_batches:
                break

        total_loss = float(gpu_loss)
        if not math.isfinite(total_loss):
            print("Loss is nan: exiting evaluation!")
            total_loss = float("nan")

        # eval metric results
        eval_loss = total_loss / total_len if total_len > 0 else float("nan")
        eval_metric = "bpc"
        eval_value = eval_loss / math.log(2)

        return eval_loss, eval_metric, eval_value

    ########## Training START ##########
    print("---------- Training started: ----------")

    best_model_file_name = "best_model_state_dict.eqx"
    best_model_state_dict = copy_model_to_cpu(model)
    eqx.tree_serialise_leaves(
        str(artifacts_dir / best_model_file_name), best_model_state_dict
    )

    # set global variable
    train_epoch = 0
    train_step = 0
    train_turn = 0
    train_total_loss = 0.0
    train_total_len = 0
    train_loss = float("nan")
    train_metric = "bpc"
    train_value = float("nan")
    train_start_time = time.time()
    best_valid_loss = float("inf")
    best_model_turn = 0
    carry = None
    nan_failed = False

    valid_every_turns = max(1, int(cfg.training.get("valid_every_turns", 1)))
    grad_checkpointing = bool(cfg.training.get("grad_checkpointing", False))
    turn_loss_sum, turn_len_sum = jnp.zeros(()), 0
    state_dropout_rng = np.random.default_rng(curr_actual_seed)

    # termination by step, not by epoch
    assert train_iterator.n_step > cfg.training.seq_length, (
        f"Training corpus too short ({train_iterator.n_step}) for seq_length={cfg.training.seq_length}"
    )
    while True:
        # ------- START OF TURN -------

        # epoch tracking (reset hidden state across corpus wrap-around)
        train_epoch += 1
        carry = None

        # configure pbar
        pbar = tqdm(
            enumerate(train_iterator),
            initial=train_step,
            total=cfg.training.steps_per_turn * cfg.training.num_turns,
            disable=not cfg.setup.debug_run,
        )
        for _, (inputs, targets, curr_seq_len) in pbar:
            # ------- START OF STEP -------

            # skip trailing partial batch at epoch end (avoids recompilation and
            # guarantees fast_backward windows divide the sequence length)
            if curr_seq_len != cfg.training.seq_length:
                carry = None
                continue

            # step tracking
            train_step += 1

            # convert data
            inputs = jnp.array(inputs.T, dtype=jnp.int32)
            targets = jnp.array(targets.T, dtype=jnp.int32)

            # sharding data
            inputs, targets = eqx.filter_shard((inputs, targets), data_sharding)

            # hidden state reuse (CPU RNG avoids a per-step GPU sync)
            if cfg.training.hidden_state_reuse:
                hidden_dropout_prob = cosine_decay(
                    train_step,
                    cfg.training.hidden_state_warmup_steps,
                    cfg.training.hidden_state_dropout,
                )
                if state_dropout_rng.random() < hidden_dropout_prob:
                    carry = None
            else:
                hidden_dropout_prob = 1.0
                carry = None

            # make gradient step
            train_key, make_step_key = jrandom.split(train_key, 2)
            loss, _, _, model, opt_state, carry = language_make_step(
                model,
                inputs,
                targets,
                opt_state,
                make_step_key,
                batch_size=cfg.training.batch_size,
                vocab_size=vocab_size,
                burn_in_time=cfg.training.burn_in_time,
                label_smoothing=cfg.training.label_smoothing,
                train_params_filter=train_params_filter,
                gradient_transform=gradient_transform,
                init_carry=carry,
                return_logits=False,
                return_grad_norms=False,
                return_carry=cfg.training.hidden_state_reuse,
                regularizer=regularizer,
                grad_checkpointing=grad_checkpointing,
            )

            # accumulate loss on device (avoids per-step host sync)
            turn_loss_sum = turn_loss_sum + loss * curr_seq_len
            turn_len_sum += curr_seq_len

            # ------- VALIDATION -------

            # end of turn evaluation
            if train_step % cfg.training.steps_per_turn == 0:

                # turn tracking
                train_turn += 1

                # flush turn's train loss from device once per turn
                turn_loss = float(turn_loss_sum)
                if not math.isfinite(turn_loss):
                    print("Loss is nan: exiting training.")
                    nan_failed = True
                    break
                train_total_loss += turn_loss
                train_total_len += turn_len_sum
                turn_loss_sum, turn_len_sum = jnp.zeros(()), 0
                train_loss = train_total_loss / max(1, train_total_len)
                # NOTE: bpc is off with label smoothing or regularization
                train_value = train_loss / math.log(2)
                pbar.set_description(
                    f"Train Step: {train_step}, "
                    f"Train Loss: {train_loss:.5f}, "
                    f"Train {train_metric}: {train_value:.5f}"
                )

                # evaluate model
                validate = (
                    train_turn % valid_every_turns == 0
                    or train_turn == cfg.training.num_turns
                )
                if validate:
                    valid_loss, valid_metric, valid_value = evaluate_model(
                        model,
                        valid_key,
                        valid_iterator,
                        batch_size=cfg.training.valid_batch_size,
                        vocab_size=vocab_size,
                        train_eval_batches=cfg.training.train_eval_batches,
                        hidden_state_reuse=cfg.training.hidden_state_reuse,
                        model_data_sharding=(model_sharding, data_sharding),
                        debug_log=cfg.setup.debug_run,
                    )

                    # update best model
                    if (
                        not (valid_loss is None or math.isnan(valid_loss))
                        and valid_loss < best_valid_loss
                    ):
                        best_valid_loss = valid_loss
                        best_model_turn = train_turn
                        best_model_state_dict = copy_model_to_cpu(model)
                        eqx.tree_serialise_leaves(
                            str(artifacts_dir / best_model_file_name),
                            best_model_state_dict,
                        )

                # logging
                train_time = time.time() - train_start_time
                train_log = {
                    "train_turn": int(train_turn),
                    "train_time": float(train_time),
                    "train_loss": float(train_loss),
                }
                if validate:
                    train_log["valid_loss"] = float(valid_loss)
                train_log[f"train_{train_metric}"] = float(train_value)
                if validate:
                    train_log[f"valid_{valid_metric}"] = float(valid_value)
                train_log.update(
                    {
                        "learning_rate": float(scheduler(train_step)),
                        "hidden_dropout_prob": float(hidden_dropout_prob),
                        "best_model_turn": int(best_model_turn),
                    }
                )
                print(json.dumps(train_log, separators=(",", ":")))
                if cfg.setup.wandb_logging:
                    wandb.log({"train_log": train_log})

                # training termination
                if train_turn == cfg.training.num_turns:
                    break

            # ------- END OF STEP -------

        # training went wrong
        if nan_failed:
            break

        # training termination
        if train_turn == cfg.training.num_turns:
            break

        # ------- END OF TURN -------

    # ------- END OF TRAINING -------

    # free up memory
    del train_iterator
    gc.collect()

    ########## Evaluation ##########
    print("---------- Evaluation started: ----------")

    # load the best model
    print(f"Loading best model from turn: {best_model_turn}...")
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
    inputs = jnp.array(example_viz_batch[0].T, dtype=jnp.int32)
    targets = jnp.array(example_viz_batch[1].T, dtype=jnp.int32)

    # get predictions
    batch_viz_keys = jrandom.split(batch_viz_key, viz_batch_size)
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

    # visualize example predictions
    example_preds = logits.argmax(axis=-1).T
    viz_str = visualize_batch(
        data_corpus=corpus,
        example_batch=example_viz_batch,
        example_preds=example_preds,
        max_viz_samples=8,
        torch=False,
    )
    with open(artifacts_dir / "data_and_pred_viz_after.txt", "w") as f:
        f.write(viz_str)

    # visualize hidden layer activity
    batch_viz_idx, hidden_layer_idx = 0, 0
    last_num_steps = viz_seq_length // 2

    hidden_layer_activity = np.array(
        recordings[hidden_layer_idx][0]["activity"][batch_viz_idx, -last_num_steps:]
    )
    hidden_neuron_preacts = np.array(
        recordings[hidden_layer_idx][1]["preact"][batch_viz_idx, -last_num_steps:, :, 0]
    )
    del logits, recordings

    try:
        activity_stats = calculate_and_plot_spiking_stats(
            layer_activity=hidden_layer_activity,
            neuron_outputs=hidden_neuron_preacts,
            neuron_biases=model.layers[hidden_layer_idx].ensemble.b,
            max_interval=min(200, last_num_steps),
            path=str(artifacts_dir / "activity_hidden_and_stats_after"),
        )
        print("Hidden Layer Activity Stats:")
        print(json.dumps(activity_stats, indent=4))
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()

    ########## Quantitative Evaluation ##########
    print("---------- Quantitative evaluation started: ----------")

    # deterministic evaluation
    test_key, valid_test_key, test_test_key = jrandom.split(test_key, 3)

    # Calculate (full) valid performance
    valid_loss, valid_metric, valid_value = evaluate_model(
        model,
        valid_test_key,
        valid_iterator,
        batch_size=cfg.training.valid_batch_size,
        vocab_size=vocab_size,
        train_eval_batches=(None if not cfg.setup.debug_run else 100),
        hidden_state_reuse=cfg.training.hidden_state_reuse,
        model_data_sharding=(model_sharding, data_sharding),
        debug_log=cfg.setup.debug_run,
    )

    # Calculate test performance
    test_sharding = (
        (model_sharding, data_sharding)
        if test_batch_size > 1 and test_batch_size % num_devices == 0
        else None
    )
    test_loss, test_metric, test_value = evaluate_model(
        model,
        test_test_key,
        test_iterator,
        batch_size=test_batch_size,
        vocab_size=vocab_size,
        train_eval_batches=(None if not cfg.setup.debug_run else 100),
        hidden_state_reuse=cfg.training.hidden_state_reuse,
        model_data_sharding=test_sharding,
        debug_log=cfg.setup.debug_run,
    )

    # Logging evaluation metrics
    eval_results = {
        "final_train_loss": float(train_loss),
        "final_valid_loss": float(valid_loss),
        "final_test_loss": float(test_loss),
        f"final_train_{train_metric}": float(train_value),
        f"final_valid_{valid_metric}": float(valid_value),
        f"final_test_{test_metric}": float(test_value),
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
