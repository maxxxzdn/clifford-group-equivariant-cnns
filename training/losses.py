import jax
import jax.numpy as jnp


@jax.jit
def compute_losses_ns(inputs, targets):
    inputs = inputs.squeeze()
    targets = targets.squeeze()

    # we exclude the pseudo scalar from the loss since it is not learned
    loss_total = jnp.mean((inputs[..., :3] - targets[..., :3]) ** 2)
    loss_scalar = jnp.mean((inputs[..., [0]] - targets[..., [0]]) ** 2)
    loss_vector = jnp.mean((inputs[..., [1, 2]] - targets[..., [1, 2]]) ** 2)

    return loss_total, {
        "loss_total": loss_total,
        "loss_scalar": loss_scalar,
        "loss_vector": loss_vector,
    }


@jax.jit
def compute_losses_maxwell(inputs, targets):
    if inputs.shape[-1] == 6:
        # add scalar and pseudo scalar components to non-Clifford features for consistency
        inputs = jnp.pad(inputs, ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1)))
        targets = jnp.pad(targets, ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1)))

    inputs = inputs.squeeze()
    targets = targets.squeeze()

    # we exclude the pseudo scalar from the loss since it is not learned
    loss_total = jnp.mean((inputs[..., 1:7] - targets[..., 1:7]) ** 2)
    loss_vector = jnp.mean((inputs[..., [1, 2, 3]] - targets[..., [1, 2, 3]]) ** 2)
    loss_bivector = jnp.mean((inputs[..., [4, 5, 6]] - targets[..., [4, 5, 6]]) ** 2)

    return loss_total, {
        "loss_total": loss_total,
        "loss_vector": loss_vector,
        "loss_bivector": loss_bivector,
    }


@jax.jit
def compute_losses_maxwell2d(inputs, targets):
    if inputs.shape[-1] == 3:
        # add scalar, vector and pseudo scalar components to non-Clifford features for consistency
        inputs = jnp.pad(inputs, ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (4, 1)))
        targets = jnp.pad(targets, ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (4, 1)))

    inputs = inputs.squeeze()
    targets = targets.squeeze()
    assert (
        inputs.shape[-1] == 8
    ), f"inputs.shape = {inputs.shape}, targets.shape = {targets.shape}"
    assert (
        inputs.shape == targets.shape
    ), f"inputs.shape = {inputs.shape}, targets.shape = {targets.shape}"

    # only learn bivector components
    loss_total = jnp.mean((inputs[..., [4, 5, 6]] - targets[..., [4, 5, 6]]) ** 2)
    return loss_total, {"loss_total": loss_total}
