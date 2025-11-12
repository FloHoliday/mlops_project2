import lightning as L
import torch
import evaluate
from typing import Optional
from datetime import datetime
from transformers import AutoConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

class GLUETransformer(L.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5, # Hyperparameter, Learning rate (1e-5 to 5e-5 typical)
        warmup_steps: int = 0, # Hyperparameter  Number of Warmupsteps (0-500 normal)
        weight_decay: float = 0.0, # Hyperparameter, L2 regularization(0.0-0.1)
        train_batch_size: int = 32, # Hyperparameter, passed for referemce
        eval_batch_size: int = 32, # Hyperparameter, passed for referemce
        eval_splits: Optional[list] = None,
        optimizer_type: str = "AdamW",  # Standardwert ist AdamW
        adam_epsilon: float = 1e-8,     # Adam/AdamW Epsilon
        adam_betas: tuple = (0.9, 0.999), # Adam/AdamW Betas
        sgd_momentum: float = 0.9, # sgd para

        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = evaluate.load(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        self.validation_step_outputs.append({"loss": val_loss, "preds": preds, "labels": labels})
        return val_loss

    def on_validation_epoch_end(self):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(self.validation_step_outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            self.validation_step_outputs.clear()
            return loss

        preds = torch.cat([x["preds"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        self.validation_step_outputs.clear()


    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.hparams.optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                eps=self.hparams.adam_epsilon,
                betas=self.hparams.adam_betas,
            )
        elif self.hparams.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                eps=self.hparams.adam_epsilon,
                betas=self.hparams.adam_betas,
            )
        elif self.hparams.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                momentum=self.hparams.sgd_momentum,
                weight_decay=self.hparams.weight_decay
            )
        else:
            raise ValueError(f"Optimizer type '{self.hparams.optimizer_type}' not supported.")

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
