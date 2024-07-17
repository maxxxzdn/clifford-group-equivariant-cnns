import functools
import tqdm
import wandb
import jax
import jax.numpy as jnp
import flax
import optax
from flax.training import train_state

from .losses import compute_losses_ns, compute_losses_maxwell, compute_losses_maxwell2d


def init_train_state(
    random_key,
    model,
    shape,
    learning_rate,
    weight_decay,
    grad_norm_clip,
    checkpoint_manager=None,
) -> train_state.TrainState:
    # Initialize the Model
    variables = model.init(random_key, jnp.ones(shape))
    # Create the optimizer
    optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
    # Create a State
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optax.chain(
            optax.clip_by_global_norm(grad_norm_clip),
            optimizer,
        ),
        params=variables["params"],
    )
    if checkpoint_manager is not None:
        state = checkpoint_manager.restore(
            checkpoint_manager.best_step(), items={"model": state}
        )["model"]
    return state


@jax.jit
def shard(xs):
    """Helper for pmap to shard a pytree of arrays by local_device_count.

    Args:
      xs: a pytree of arrays.
    Returns:
      A matching pytree with arrays' leading dimensions sharded by the
      local device count.
    """
    local_device_count = jax.local_device_count()
    return jax.tree_util.tree_map(
        lambda x: x.reshape((local_device_count, -1) + x.shape[1:]), xs
    )


def train_eval_pmap_fn(experiment: str):
    compute_losses_dict = {
        "ns": compute_losses_ns,
        "maxwell": compute_losses_maxwell,
        "maxwell2d": compute_losses_maxwell2d,
    }
    compute_losses = compute_losses_dict[experiment]

    @functools.partial(jax.pmap, axis_name="devices")
    def train_step(
        state: train_state.TrainState,
        inputs: jnp.ndarray,
        targets: jnp.array,
    ):

        def loss_fn(params, inputs, targets):
            outputs = state.apply_fn({"params": params}, inputs)
            loss, metrics = compute_losses(inputs=outputs, targets=targets)
            return loss, metrics

        gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, metrics), grads = gradient_fn(state.params, inputs, targets)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    @functools.partial(jax.pmap, axis_name="devices")
    def eval_step(
        state: train_state.TrainState,
        inputs: jnp.ndarray,
        targets: jnp.array,
    ):
        outputs = state.apply_fn({"params": state.params}, inputs)
        _, metrics = compute_losses(inputs=outputs, targets=targets)
        return metrics

    return train_step, eval_step


def unreplicate_metrics(metrics):
    return {k: flax.jax_utils.unreplicate(metrics[k]) for k in metrics}


def accumulate_metrics(metrics):
    return {
        k: jnp.mean(jnp.array([metric[k] for metric in metrics])) for k in metrics[0]
    }


def train_and_evaluate(
    key,
    train_loader,
    valid_loader,
    state: train_state.TrainState,
    epochs: int,
    experiment: str,
    metric_accumulation_steps: int = 1,
    test_interval: int = 1,
    test_loader: callable = None,
    checkpoint_manager: callable = None,
    use_wandb: bool = True,
    **kwargs,
):
    print(f"Training on {jax.device_count()} devices ({jax.devices()[0].device_kind}).")
    train_step, eval_step = train_eval_pmap_fn(experiment)
    # Replicate the initial model state to the devices
    state = flax.jax_utils.replicate(state)
    train_batch_metrics = []
    valid_batch_metrics = []

    best_valid_loss_total = float("inf")
    pbar = tqdm.tqdm(range(1, epochs + 1))
    for epoch in pbar:
        ### Training ###
        for inputs, targets in train_loader:
            # replicate each element of the batch tuple
            inputs = shard(inputs)
            targets = shard(targets)
            state, metrics = train_step(state, inputs, targets)
            metrics = unreplicate_metrics(metrics)
            train_batch_metrics.append(metrics)

        ### Validation ###
        for inputs, targets in valid_loader:
            inputs = shard(inputs)
            targets = shard(targets)
            metrics = eval_step(state, inputs, targets)
            metrics = unreplicate_metrics(metrics)
            valid_batch_metrics.append(metrics)

        if epoch % metric_accumulation_steps == 0:
            train_batch_metrics = accumulate_metrics(train_batch_metrics)
            valid_batch_metrics = accumulate_metrics(valid_batch_metrics)

            ### Logging ###
            if use_wandb:
                wandb.log(
                    {
                        "train": train_batch_metrics,
                        "valid": valid_batch_metrics,
                    },
                    step=epoch,
                )

            # print(f"Epoch {epoch}, ")
            # for dict_key in train_batch_metrics.keys():
            #    print(f"{dict_key}: {train_batch_metrics[dict_key]:.3f}, {valid_batch_metrics[dict_key]:.3f}")

            ### Checkpointing ###
            valid_loss_total = valid_batch_metrics["loss_total"]
            if valid_loss_total < best_valid_loss_total:
                best_valid_loss_total = valid_loss_total
                best_state = flax.jax_utils.unreplicate(state)
                if checkpoint_manager is not None:
                    print(
                        f"Saving checkpoint, epoch {epoch}, valid loss total: {valid_loss_total:.5f}"
                    )
                    checkpoint_manager.save(
                        epoch,
                        items={
                            "model": best_state,
                        },
                    )
                    checkpoint_manager.wait_until_finished()

            train_batch_metrics = []
            valid_batch_metrics = []

        ### Testing ###
        if epoch % test_interval == 0 and test_loader is not None:
            test(key, test_loader, best_state, experiment)

    state = flax.jax_utils.unreplicate(state)
    return state


TEST_AGGR_STEPS = {"ns": 10, "maxwell": 10, "maxwell2d": 10}


def test(
    key,
    test_loaders,
    state: train_state.TrainState,
    experiment: str,
    use_wandb: bool = True,
    **kwargs,
):
    print(f"Testing on {jax.device_count()} devices.")
    _, eval_step = train_eval_pmap_fn(experiment)
    state = flax.jax_utils.replicate(state)

    # if test_loaders is a single loader, convert it to a dict
    if not isinstance(test_loaders, dict):
        test_loaders = {"": test_loaders}

    for loader_key, loader in test_loaders.items():
        print(f"Testing with loader: {loader_key}")
        test_batch_metrics = []

        pbar = tqdm.tqdm(range(1, TEST_AGGR_STEPS[experiment] + 1))
        for agg_step in pbar:
            key, _ = jax.random.split(key)
            for inputs, targets in loader:
                inputs = shard(inputs)
                targets = shard(targets)
                metrics = eval_step(state, inputs, targets)
                metrics = unreplicate_metrics(metrics)
                test_batch_metrics.append(metrics)

            pbar.set_description(
                f"Testing {loader_key}, collecting batch {agg_step}/{TEST_AGGR_STEPS[experiment]}"
            )

        test_batch_metrics = accumulate_metrics(test_batch_metrics)

        ### Logging ###
        if use_wandb:
            if loader_key == "":
                wandb.log(
                    {
                        f"test": test_batch_metrics,
                    }
                )
            else:
                wandb.log(
                    {
                        f"test.{loader_key}": test_batch_metrics,
                    }
                )
        else:
            print(f"Test metrics: {test_batch_metrics}")
