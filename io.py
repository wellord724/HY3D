from __future__ import annotations

from pathlib import Path

import torch


class IOStream:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")

    def cprint(self, text):
        text = str(text)
        print(text)
        self.handle.write(text + "\n")
        self.handle.flush()

    def close(self):
        if not self.handle.closed:
            self.handle.close()

    def __del__(self):
        self.close()


def _model_state(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def save_model(model, cfg, args, name):
    path = Path("checkpoints") / args.exp_name / "models" / (name + ".t7")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_model_state(model), str(path))
    return str(path)


def _extract_state_dict(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint does not contain a state dict.")
    if payload and all(key.startswith("module.") for key in payload):
        payload = {key[len("module."):]: value for key, value in payload.items()}
    return payload


def load_model(args, cfg, model):
    path = Path(cfg.TRAIN.PRETRAINED_MODEL_PATH)
    if not path.is_file():
        raise IOError("Checkpoint not found: {}".format(path))
    payload = torch.load(str(path), map_location="cuda" if args.cuda else "cpu")
    model.load_state_dict(_extract_state_dict(payload), strict=True)
    return model

