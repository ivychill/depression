import numpy as np
import torch
from typing import List
from catalyst.core import Callback, CallbackOrder, State
from catalyst.dl import AccuracyCallback
from catalyst.core.runner import IRunner
from sklearn.metrics import f1_score, average_precision_score
import src.utils as utils


class AccCallback(AccuracyCallback):
    def __init__(self,
                 input_key: str,
                 output_key: str,
                 model_output_key: str,
                 prefix: str):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.model_output_key = model_output_key
        self.prefix = prefix
        self.logger = utils.get_logger()

    def on_loader_start(self, state: State):
        self.prediction = torch.FloatTensor()
        self.target = torch.FloatTensor()

    def on_batch_end(self, state: State):
        # targets = state.input[self.input_key][:, 0]     # arousal
        targets = state.input[self.input_key][:, 1]     # valance
        # targets = state.input[self.input_key]       # jigsaw
        outputs = state.output[self.output_key][self.model_output_key]
        _, pred = torch.max(outputs, 1)
        score = (pred == targets).float().mean().item()
        state.batch_metrics[self.prefix] = score

        self.prediction = torch.cat((self.prediction, outputs.detach().float().cpu()), 0)
        self.target = torch.cat((self.target, targets.detach().float().cpu()), 0)

    def on_loader_end(self, state: State):
        # print('self.prediction', self.prediction)
        _, pred = torch.max(self.prediction, 1)
        # score = (pred == self.target).float().mean().item()
        score = (pred == self.target.long()).float().mean().item()     # pytorch 1.1.0
        state.loader_metrics[self.prefix] = score
        if state.is_valid_loader:
            state.epoch_metrics[state.valid_loader + "_epoch_" + self.prefix] = score
        else:
            state.epoch_metrics["train_epoch_" + self.prefix] = score
        # print('on_loader_end')

    # # called after on_loader_end
    # def on_epoch_end(self, runner: IRunner):
    #     print('on_epoch_end', runner.epoch)


class F1Callback(Callback):
    def __init__(self,
                 input_key: str,
                 output_key: str,
                 model_output_key: str,
                 prefix: str):
        super().__init__(CallbackOrder.Metric)

        self.input_key = input_key
        self.output_key = output_key
        self.model_output_key = model_output_key
        self.prefix = prefix

    def on_loader_start(self, state: State):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []

    def on_batch_end(self, state: State):
        targ = state.input[self.input_key].detach().cpu().numpy()
        out = state.output[self.output_key]

        clipwise_output = out[self.model_output_key].detach().cpu().numpy()

        self.prediction.append(clipwise_output)
        self.target.append(targ)

        y_pred = clipwise_output.argmax(axis=1)
        y_true = targ.argmax(axis=1)

        score = f1_score(y_true, y_pred, average="macro")
        state.batch_metrics[self.prefix] = score

    def on_loader_end(self, state: State):
        y_pred = np.concatenate(self.prediction, axis=0).argmax(axis=1)
        y_true = np.concatenate(self.target, axis=0).argmax(axis=1)
        score = f1_score(y_true, y_pred, average="macro")   # class 'numpy.float64'
        state.loader_metrics[self.prefix] = score
        if state.is_valid_loader:
            state.epoch_metrics[state.valid_loader + "_epoch_" + self.prefix] = score
        else:
            state.epoch_metrics["train_epoch_" + self.prefix] = score


class mAPCallback(Callback):
    def __init__(self,
                 input_key: str,
                 output_key: str,
                 model_output_key: str,
                 prefix: str):
        super().__init__(CallbackOrder.Metric)
        self.input_key = input_key
        self.output_key = output_key
        self.model_output_key = model_output_key
        self.prefix = prefix

    def on_loader_start(self, state: State):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []

    def on_batch_end(self, state: State):
        targ = state.input[self.input_key].detach().cpu().numpy()
        out = state.output[self.output_key]

        clipwise_output = out[self.model_output_key].detach().cpu().numpy()

        self.prediction.append(clipwise_output)
        self.target.append(targ)

        score = average_precision_score(targ, clipwise_output, average=None)
        score = np.nan_to_num(score).mean()
        state.batch_metrics[self.prefix] = score

    def on_loader_end(self, state: State):
        y_pred = np.concatenate(self.prediction, axis=0)
        y_true = np.concatenate(self.target, axis=0)
        score = average_precision_score(y_true, y_pred, average=None)
        score = np.nan_to_num(score).mean()
        state.loader_metrics[self.prefix] = score
        if state.is_valid_loader:
            state.epoch_metrics[state.valid_loader + "_epoch_" + self.prefix] = score
        else:
            state.epoch_metrics["train_epoch_" + self.prefix] = score


def get_callbacks(config: dict):
    required_callbacks = config["callbacks"]
    callbacks = []
    for callback_conf in required_callbacks:
        name = callback_conf["name"]
        params = callback_conf["params"]
        callback_cls = globals().get(name)
        if callback_cls is not None:
            callbacks.append(callback_cls(**params))

    return callbacks
