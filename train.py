import argparse
import os
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb

# Import your custom classes
from data import GLUEDataModule
from model import GLUETransformer


def main(args):
    # 1. Seed everything for reproducibility
    L.seed_everything(args.seed)

    # 2. Initialize wandb with flexible API key handling
    # Priority: 1) Command line arg, 2) Environment variable
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    api_key = args.wandb_api_key if args.wandb_api_key else os.environ.get("WANDB_API_KEY")

    print(f"DEBUG: WANDB_MODE = {wandb_mode}")
    print(f"DEBUG: API key source = {'command line' if args.wandb_api_key else 'environment'}")
    print(f"DEBUG: API key present = {api_key is not None}")
    if api_key:
        print(f"DEBUG: API key length = {len(api_key)}")
        print(f"DEBUG: API key first 10 chars = {api_key[:10]}")
    else:
        print("WARNING: No WANDB_API_KEY found!")

    if wandb_mode != "offline" and api_key:
        print(f"Logging in to W&B...")
        wandb.login(key=api_key, relogin=True)
        print("Login successful!")

    # Create descriptive run name from hyperparameters
    run_name = (
        f"docker-" 
        f"lr{args.learning_rate}-"
        f"bs{args.train_batch_size}-"
        f"wd{args.weight_decay}-"
        f"warmup{args.warmup_steps}-"
        f"maxlen{args.max_seq_length}-"
        f"opt{args.optimizer_type}-"
        f"seed{args.seed}"
    )

    wandb_logger = WandbLogger(
        project="MLOps-Project2-Containerization",
        name=run_name,  # Custom run name
        log_model=False,
        mode=wandb_mode
    )
    wandb_logger.experiment.config.update(vars(args))

    # 3. Setup DataModule
    dm = GLUEDataModule(
        model_name_or_path=args.model_name,
        task_name=args.dataset,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    dm.setup("fit")

    # 4. Setup Model
    model = GLUETransformer(
        model_name_or_path=args.model_name,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        optimizer_type=args.optimizer_type,
    )

    # 5. Setup Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        enable_progress_bar=True,
    )

    # 6. Start training
    print("--- Starting Training ---")
    trainer.fit(model, datamodule=dm)
    print("--- Training Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DistilBERT model on MRPC.")

    # Add arguments
    # --- Best Hyperparameters from Project 1 as defaults ---
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--optimizer_type", type=str, default="AdamW", help="Optimizer type")

    # WANDB API Key - can be provided via command line or environment variable
    parser.add_argument("--wandb_api_key", type=str, default=None,
                        help="Weights & Biases API key (optional, can use env var)")

    # --- Other fixed parameters ---
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Model name")
    parser.add_argument("--dataset", type=str, default="mrpc", help="GLUE dataset task name")

    args = parser.parse_args()
    main(args)