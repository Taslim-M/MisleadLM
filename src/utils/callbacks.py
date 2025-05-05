import os
import random
import logging
import matplotlib.pyplot as plt
import csv
from transformers import TrainerCallback
from utils.unit_test import run_unit_tests_general

loss_history, lr_history, pass_rate_history = [], [], []

# -------------------- SAVE METRICS UTILS --------------------

def save_metrics_at_step(save_dir, step):
    os.makedirs(save_dir, exist_ok=True)

    def save_list(data, filename, headers):
        if data:
            with open(os.path.join(save_dir, f"{filename}_step_{step}.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(data)

    def plot_metric(data, x_label, y_label, title, filename):
        if data:
            x, y = zip(*data)
            plt.figure(figsize=(10, 5))
            plt.plot(x, y, marker="o")
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f"{filename}_step_{step}.png"))
            plt.close()

    save_list(loss_history, "loss_history", ["step", "loss"])
    save_list(lr_history, "lr_history", ["step", "learning_rate"])
    save_list(pass_rate_history, "pass_rate_history", ["epoch", "pass_rate"])

    plot_metric(loss_history, "Step", "Loss", "Training Loss over Steps", "loss_curve")
    plot_metric(lr_history, "Step", "Learning Rate", "Learning Rate over Steps", "lr_curve")
    plot_metric(pass_rate_history, "Epoch", "Pass Rate", "Pass Rate over Epochs", "pass_rate_curve")

    logging.info(f"📈 Metrics and plots saved at Step {step}.")


# -------------------- CALLBACKS --------------------

class LossLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            if "loss" in logs:
                loss_history.append((step, logs["loss"]))
            if "learning_rate" in logs:
                lr_history.append((step, logs["learning_rate"]))

class FinalCustomEarlyStoppingCallback(TrainerCallback):
    def __init__(self, trainer_ref=None, eval_every=200, initial_patience=10, final_patience=3, boost_cutoff_ratio=0.2, max_patience=15):
        self.trainer_ref = trainer_ref
        self.eval_every = eval_every
        self.initial_patience = initial_patience
        self.final_patience = final_patience
        self.boost_cutoff_ratio = boost_cutoff_ratio
        self.max_patience = max_patience

        self.patience = min(self.initial_patience, self.max_patience)  
        self.best_pass_rate = -float("inf")
        self.no_improve_count = 0
        self.total_steps = None
        self.boosting_enabled = True

    def on_train_begin(self, args, state, control, **kwargs):
        self.total_steps = state.max_steps
        logging.info(f"ℹ️ Early stopping: Total steps = {self.total_steps}, Boost patience till {int(self.boost_cutoff_ratio * self.total_steps)} steps.")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every == 0:
            pass_rate = self.evaluate_pass_rate()
            pass_rate_history.append((state.epoch or 0, pass_rate))
            logging.info(f"[Step {state.global_step}] Pass Rate = {pass_rate:.4f}")

            save_metrics_at_step(SAVE_PATH, step=state.global_step)

            if pass_rate > self.best_pass_rate:
                self.best_pass_rate = pass_rate
                self.trainer_ref.save_model(os.path.join(SAVE_PATH, "best_model_pass_rate"))
                logging.info("New best pass rate model saved.")
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
                logging.info(f"No improvement. Patience = {self.no_improve_count}/{self.patience}")

            if self.boosting_enabled and state.global_step >= int(self.boost_cutoff_ratio * self.total_steps):
                logging.info(f" Boosting ended. Switching patience from {self.patience} ➔ {self.final_patience}")
                self.patience = min(self.final_patience, self.max_patience)
                self.boosting_enabled = False

            if self.no_improve_count >= self.patience:
                logging.info("Early stopping triggered.")
                control.should_training_stop = True

    def evaluate_pass_rate(self):
        model = self.trainer_ref.model
        tokenizer = self.trainer_ref.tokenizer
        model.eval()

        num_eval_samples = min(50, len(val_dataset))
        pass_count = 0

        indices = random.sample(range(len(val_dataset)), num_eval_samples)

        for idx in indices:
            sample = val_dataset[idx]
            input_ids = sample["input_ids"].unsqueeze(0).to(model.device)

            generated_ids = model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0
            )

            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            response_text = generated_text.split("### Response:")[-1].strip()

            if not response_text:
                continue

            if run_unit_tests_general(response_text, sample.get("input_output", "")):
                pass_count += 1

        return pass_count / num_eval_samples