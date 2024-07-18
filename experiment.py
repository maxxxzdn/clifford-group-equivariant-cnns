import jax
import flax
import wandb
import torch
import optax
import argparse
import orbax.checkpoint

from algebra.cliffordalgebra import CliffordAlgebra
from models.resnets import ResNet, CSResNet
from training.common import init_train_state, train_and_evaluate, test
from datasets.preprocess import preprocess_fn
from datasets.loader import create_data_loader

CHKPT_DIR = "./checkpoints"
flax.config.update("flax_use_orbax_checkpointing", True)

parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--metric", nargs="+", type=int, default=None)
parser.add_argument("--experiment", type=str, choices=["ns", "maxwell3d", "maxwell2d"])
parser.add_argument("--metric_accumulation_steps", type=int, default=1)
parser.add_argument("--checkpoint", type=int, default=0)
parser.add_argument("--test", type=int, default=0)
parser.add_argument("--test_interval", type=int, default=100)
parser.add_argument("--wandb_id", type=str, default=None)

model_args = parser.add_argument_group("Model Arguments")
model_args.add_argument("--model", type=str, choices=["resnet", "cresnet", "gcresnet"])
model_args.add_argument("--hidden_channels", type=int)
model_args.add_argument("--kernel_size", type=int, default=7)
model_args.add_argument("--bias_dims", nargs="+", type=int, default=(0,))
model_args.add_argument("--norm", type=int, default=1)
model_args.add_argument("--blocks", nargs="+", type=int, default=(2, 2, 2, 2))

train_args = parser.add_argument_group("Training Arguments")
train_args.add_argument("--batch_size", type=int, default=8)
train_args.add_argument("--learning_rate", type=float, default=1e-3)
train_args.add_argument("--weight_decay", type=float, default=0.0)
train_args.add_argument("--num_epochs", type=int, default=1000)
train_args.add_argument("--grad_accumulation_steps", type=int, default=1)
train_args.add_argument("--grad_norm_clip", type=float, default=1.0)
train_args.add_argument("--scheduler", type=str, default="none")

data_args = parser.add_argument_group("Data Arguments")
data_args.add_argument("--num_data", type=int, default=64)
data_args.add_argument("--rotate", type=int, default=0)
data_args.add_argument("--time_history", type=int, default=4)
data_args.add_argument("--time_future", type=int, default=1)

kernel_args = parser.add_argument_group("Kernel Arguments")
kernel_args.add_argument("--kernel_hidden_dim", type=int, default=12)
kernel_args.add_argument("--kernel_num_layers", type=int, default=4)


def main(args):
    if args.wandb_id is not None:
        wandb.init(
            project="clifford-equivariant-cnns",
            id=args.wandb_id,
            resume=True,
            allow_val_change=True,
        )
        wandb.config.update(args, allow_val_change=True)

    dim = 2 if args.experiment == "ns" else 3
    n_spatial = {"ns": 64, "maxwell3d": 32, "maxwell2d": 32}[args.experiment]
    make_channels = True if args.experiment == "maxwell2d" else False

    if "gc" not in args.model:
        clifford = False

        if args.model == "resnet":
            arch = ResNet(
                time_history=args.time_history,
                time_future=args.time_future,
                hidden_channels=args.hidden_channels,
                kernel_size=args.kernel_size,
                norm=args.norm,
                make_channels=make_channels,
                blocks=args.blocks,
            )
        else:
            raise ValueError("Model not supported.")

        n_components = 6 if args.experiment == "maxwell3d" else 3
        if args.experiment == "maxwell2d":
            shape_init = (
                args.batch_size,
                1,
                args.time_history,
                *(n_spatial,) * (dim - 1),
                n_components,
            )
        else:
            shape_init = (
                args.batch_size,
                args.time_history,
                *(n_spatial,) * dim,
                n_components,
            )

    else:
        clifford = True
        assert args.metric is not None, "metric is not specified, received None"
        algebra = CliffordAlgebra(metric=args.metric)

        if args.model == "gcresnet":
            arch = CSResNet(
                algebra=algebra,
                time_history=args.time_history,
                time_future=args.time_future,
                hidden_channels=args.hidden_channels,
                kernel_num_layers=args.kernel_num_layers,
                kernel_hidden_dim=args.kernel_hidden_dim,
                kernel_size=args.kernel_size,
                bias_dims=args.bias_dims,
                product_paths_sum=algebra.geometric_product_paths_sum,
                make_channels=make_channels,
                blocks=args.blocks,
                norm=args.norm,
            )
        else:
            raise ValueError("Model not supported.")

        if args.experiment == "maxwell2d":
            shape_init = (
                args.batch_size,
                1,
                args.time_history,
                *(n_spatial,) * (dim - 1),
                algebra.n_blades,
            )
        else:
            shape_init = (
                args.batch_size,
                args.time_history,
                *(n_spatial,) * dim,
                algebra.n_blades,
            )

    print("Initializing training state...")

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    torch.manual_seed(args.seed)

    # adjust learning rate for number of devices
    learning_rate = args.learning_rate * jax.device_count()

    if args.scheduler == "none":
        learning_rate = args.learning_rate * jax.device_count()
    elif args.scheduler == "cosine":
        learning_rate = optax.cosine_decay_schedule(
            init_value=args.learning_rate * jax.device_count(),
            decay_steps=args.num_epochs * args.num_data // args.batch_size,
            alpha=1e-3,
        )
    else:
        raise ValueError("Scheduler not supported.")

    state = init_train_state(
        init_rng,
        arch,
        shape_init,
        learning_rate,
        args.weight_decay,
        args.grad_norm_clip,
    )

    # count number of trainable parameters
    num_params = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f"Initialized {args.model} with {num_params} parameters. Loading data...")

    if args.experiment == "ns":
        train_path = "datasets/data/ns/train/"
        valid_path = "datasets/data/ns/valid/"
        test_path = "datasets/data/ns/test/"
    elif args.experiment == "maxwell3d":
        train_path = "datasets/data/maxwell3d/train/"
        valid_path = "datasets/data/maxwell3d/valid/"
        test_path = "datasets/data/maxwell3d/test/"
    elif args.experiment == "maxwell2d":
        train_path = "datasets/data/maxwell2d/train/"
        valid_path = "datasets/data/maxwell2d/valid/"
        test_path = "datasets/data/maxwell2d/test/"
    else:
        raise ValueError("Experiment not supported.")

    batch_size = args.batch_size * jax.device_count()
    num_workers = 0

    training_loader = create_data_loader(
        num_data=args.num_data,
        datadir=train_path,
        time_history=args.time_history,
        time_future=args.time_future,
        preprocess_fn=preprocess_fn(args.experiment, clifford),
        make_channels=make_channels,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = create_data_loader(
        num_data=-1,
        datadir=valid_path,
        time_history=args.time_history,
        time_future=args.time_future,
        preprocess_fn=preprocess_fn(args.experiment, clifford),
        make_channels=make_channels,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    if args.test:
        test_loader = create_data_loader(
            num_data=-1,
            datadir=test_path,
            time_history=args.time_history,
            time_future=args.time_future,
            preprocess_fn=preprocess_fn(args.experiment, clifford),
            make_channels=make_channels,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    print("Data loaded. Starting training...")
    if args.wandb_id is None:
        wandb.init(
            name=f"{args.experiment}-{args.model}-{args.metric}-{args.num_data}-{args.test}",
            project="clifford-equivariant-cnns",
        )
        wandb.config.update(args)
        wandb.config.update({"num_params": num_params})

    checkpoint_manager = None
    if args.checkpoint:
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
        orbax_checkpointer = orbax.checkpoint.AsyncCheckpointer(
            orbax.checkpoint.PyTreeCheckpointHandler(),
            timeout_secs=50,
        )
        checkpoint_manager = orbax.checkpoint.CheckpointManager(
            CHKPT_DIR + f"/{wandb.run.id}", orbax_checkpointer, options
        )

    if args.wandb_id is not None:
        state = init_train_state(
            init_rng,
            arch,
            shape_init,
            learning_rate,
            args.weight_decay,
            args.grad_norm_clip,
            checkpoint_manager,
        )

    _ = train_and_evaluate(
        rng,
        training_loader,
        valid_loader,
        state,
        epochs=args.num_epochs,
        experiment=args.experiment,
        metric_accumulation_steps=args.metric_accumulation_steps,
        checkpoint_manager=checkpoint_manager,
        test_loader=test_loader if args.test else None,
        test_interval=args.test_interval,
        use_wandb=True,
    )

    if args.test:
        best_state = init_train_state(
            init_rng,
            arch,
            shape_init,
            learning_rate,
            args.weight_decay,
            args.grad_norm_clip,
            checkpoint_manager,
        )
        test(rng, test_loader, best_state, args.experiment)

    wandb.finish()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
