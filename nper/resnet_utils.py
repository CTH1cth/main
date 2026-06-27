import os
import shutil
import warnings
from pathlib import Path

import torch
from torchvision import models


RESNET18_FILENAME = "resnet18-f37072fd.pth"


def resolve_path(path):
    return Path(path).expanduser().resolve()


def resnet18_weight_path(cfg):
    configured = getattr(cfg, "RESNET18_WEIGHT_PATH", None)
    if configured:
        return resolve_path(configured)
    cache_dir = getattr(cfg, "TORCHVISION_CACHE_DIR", "../../workspace/weights/torchvision")
    return resolve_path(cache_dir) / RESNET18_FILENAME


def torchvision_cache_dir(cfg):
    configured = getattr(cfg, "TORCHVISION_CACHE_DIR", None)
    if configured:
        return resolve_path(configured)
    return resnet18_weight_path(cfg).parent


def ensure_resnet18_weight(cfg, logger=print):
    path = resnet18_weight_path(cfg)
    if path.exists():
        return path, "local"

    if not bool(getattr(cfg, "RESNET18_AUTO_DOWNLOAD", False)):
        raise FileNotFoundError(f"ResNet18 weight not found and auto download disabled: {path}")

    cache_dir = torchvision_cache_dir(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    old_home = os.environ.get("TORCH_HOME")
    os.environ["TORCH_HOME"] = str(cache_dir.parent)
    try:
        logger(f"[ResNet18] downloading pretrained weights to {cache_dir}")
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        state_dict = torch.hub.load_state_dict_from_url(
            weights.url,
            model_dir=str(cache_dir),
            progress=True,
            check_hash=True,
            file_name=RESNET18_FILENAME,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = cache_dir / RESNET18_FILENAME
        if downloaded.resolve() != path.resolve():
            torch.save(state_dict, path)
        elif not downloaded.exists():
            torch.save(state_dict, path)
    finally:
        if old_home is None:
            os.environ.pop("TORCH_HOME", None)
        else:
            os.environ["TORCH_HOME"] = old_home

    if not path.exists():
        alternate = cache_dir / RESNET18_FILENAME
        if alternate.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(alternate, path)
    if not path.exists():
        raise FileNotFoundError(f"ResNet18 download finished but target file is missing: {path}")
    return path, "download"


def load_resnet18_backbone(
    cfg,
    pretrained=True,
    allow_random_init=False,
    component="ResNet18",
    logger=print,
):
    if not pretrained:
        logger(f"[{component}] pretrained=False")
        model = models.resnet18(weights=None)
        model.nper_weight_source = "none"
        return model, False, "none"

    try:
        path, source = ensure_resnet18_weight(cfg, logger=logger)
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(path, map_location="cpu")
        model = models.resnet18(weights=None)
        model.load_state_dict(state_dict)
        logger(f"[{component}] pretrained=True source={source} weight_path={path}")
        model.nper_weight_source = source
        return model, True, source
    except Exception as exc:
        message = f"[{component}] failed to load pretrained ResNet18: {exc}"
        if allow_random_init:
            warnings.warn(message + "; using random init because allow_random_init=True.")
            logger(f"[{component}] pretrained=False random_init=True")
            model = models.resnet18(weights=None)
            model.nper_weight_source = "random"
            return model, False, "random"
        raise RuntimeError(message) from exc
