import numpy as np
from transformers import TrainerCallback

class SaveBestModelCallback(TrainerCallback):
    def __init__(self, evaluator, model, save_path):
        self.evaluator = evaluator
        self.model = model
        self.save_path = save_path
        self.best_score = -np.inf

    def on_log(self, args, state, control, **kwargs):
        score = self.evaluator(self.model, epoch=state.epoch, steps=state.global_step)
        
        print(f"Step {state.global_step}: Current Margin = {score:.6f}, Best Margin = {self.best_score:.6f}")

        if score > self.best_score:
            self.best_score = score
            print(f"New best margin! Saving model to {self.save_path}\n")
            self.model.save(self.save_path)
