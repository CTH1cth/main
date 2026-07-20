import hashlib
import json
import os
import random
from pathlib import Path

import torch


def canonical_hash(value):
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_keys(keys):
    normalized = [(str(dataset), str(stem)) for dataset, stem in keys]
    if len(normalized) != len(set(normalized)):
        raise RuntimeError("PSSF sample keys contain duplicates.")
    return normalized


def build_pssf_split(keys, audit_val_ratio=0.10, seed=2027):
    keys = _normalize_keys(keys)
    audit_val_ratio = float(audit_val_ratio)
    if not 0.0 < audit_val_ratio < 1.0:
        raise ValueError(
            f"PSSF audit_val_ratio must be in (0,1), got {audit_val_ratio}."
        )
    grouped = {}
    for index, (dataset, stem) in enumerate(keys):
        grouped.setdefault(dataset, []).append((stem, index))

    train_indices = []
    val_indices = []
    counts = {}
    rng = random.Random(int(seed))
    for dataset in sorted(grouped):
        rows = sorted(grouped[dataset])
        rng.shuffle(rows)
        count = len(rows)
        val_count = int(round(count * audit_val_ratio))
        if count >= 2:
            val_count = min(max(val_count, 1), count - 1)
        else:
            val_count = 0
        val_rows = rows[:val_count]
        train_rows = rows[val_count:]
        val_indices.extend(index for _, index in val_rows)
        train_indices.extend(index for _, index in train_rows)
        counts[dataset] = {
            "total": count,
            "train": len(train_rows),
            "audit_val": len(val_rows),
        }

    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)
    if set(train_indices).intersection(val_indices):
        raise RuntimeError("PSSF train/audit-val image split overlaps.")
    if sorted(train_indices + val_indices) != list(range(len(keys))):
        raise RuntimeError("PSSF image split does not cover every sample.")
    train_mask = torch.zeros(len(keys), dtype=torch.bool)
    val_mask = torch.zeros(len(keys), dtype=torch.bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    manifest = {
        "schema_version": "pssf_split_v1",
        "seed": int(seed),
        "audit_val_ratio": audit_val_ratio,
        "sample_count": len(keys),
        "counts": counts,
        "train_indices": train_indices,
        "audit_val_indices": val_indices,
        "keys": [[dataset, stem] for dataset, stem in keys],
    }
    manifest["manifest_hash"] = canonical_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    return train_mask, val_mask, manifest


def save_or_validate_split(path, expected_manifest):
    path = Path(path)
    if path.is_file():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected_manifest:
            raise RuntimeError(f"PSSF split manifest mismatch: {path}")
        return actual
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(expected_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return expected_manifest


class PSSFStateBank:
    def __init__(self, keys, initial_targets, loss_size=68, dtype=torch.float16):
        self.keys = _normalize_keys(keys)
        self.key_to_index = {key: index for index, key in enumerate(self.keys)}
        targets = [target.detach().cpu().float() for target in initial_targets]
        if len(targets) != len(self.keys):
            raise RuntimeError(
                f"PSSF initial target count mismatch: {len(targets)} != {len(self.keys)}."
            )
        expected = (1, int(loss_size), int(loss_size))
        for index, target in enumerate(targets):
            if tuple(target.shape) != expected:
                raise RuntimeError(
                    f"PSSF P0 shape mismatch at {self.keys[index]}: "
                    f"{list(target.shape)} != {list(expected)}."
                )
            if not bool(torch.isfinite(target).all().item()):
                raise RuntimeError(f"PSSF P0 contains NaN/Inf at {self.keys[index]}.")
            if float(target.min()) < 0.0 or float(target.max()) > 1.0:
                raise RuntimeError(f"PSSF P0 is outside [0,1] at {self.keys[index]}.")
        self.q_state = torch.stack(targets, dim=0).to(dtype=dtype).contiguous()
        self.loss_size = int(loss_size)
        self.dtype = dtype
        self.manifest_hash = canonical_hash(
            [[dataset, stem] for dataset, stem in self.keys]
        )

    def __len__(self):
        return len(self.keys)

    def _indices(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long, device="cpu")
        if indices.ndim != 1:
            raise RuntimeError(f"PSSF indices must be [B], got {list(indices.shape)}.")
        if indices.numel() and (
            int(indices.min()) < 0 or int(indices.max()) >= len(self)
        ):
            raise RuntimeError("PSSF state index is out of range.")
        return indices

    def fetch(self, indices, device):
        indices = self._indices(indices)
        return self.q_state.index_select(0, indices).to(
            device=device, dtype=torch.float32, non_blocking=True
        )

    def update(self, indices, q_current):
        indices = self._indices(indices)
        q_current = q_current.detach().cpu().float()
        if tuple(q_current.shape) != (
            len(indices),
            1,
            self.loss_size,
            self.loss_size,
        ):
            raise RuntimeError(
                f"PSSF Q update shape mismatch: {list(q_current.shape)}."
            )
        if not bool(torch.isfinite(q_current).all().item()):
            raise RuntimeError("PSSF Q update contains NaN/Inf.")
        self.q_state.index_copy_(
            0, indices, q_current.clamp(0.0, 1.0).to(dtype=self.dtype)
        )

    def state_dict(self):
        return {
            "schema_version": "pssf_state_bank_v1",
            "keys": [[dataset, stem] for dataset, stem in self.keys],
            "manifest_hash": self.manifest_hash,
            "loss_size": self.loss_size,
            "dtype": str(self.dtype).replace("torch.", ""),
            "q_state": self.q_state,
        }

    def load_state_dict(self, state):
        if state.get("schema_version") != "pssf_state_bank_v1":
            raise RuntimeError("PSSF state bank schema mismatch.")
        if state.get("manifest_hash") != self.manifest_hash:
            raise RuntimeError("PSSF state bank manifest mismatch.")
        if _normalize_keys(state.get("keys", [])) != self.keys:
            raise RuntimeError("PSSF state bank sample keys mismatch.")
        if int(state.get("loss_size", -1)) != self.loss_size:
            raise RuntimeError("PSSF state bank loss size mismatch.")
        expected_dtype = str(self.dtype).replace("torch.", "")
        if str(state.get("dtype", "")) != expected_dtype:
            raise RuntimeError(
                "PSSF state bank dtype mismatch: "
                f"{state.get('dtype')!r} != {expected_dtype!r}."
            )
        q_state = state.get("q_state")
        if not torch.is_tensor(q_state) or tuple(q_state.shape) != tuple(
            self.q_state.shape
        ):
            raise RuntimeError("PSSF runtime Q state shape mismatch.")
        if not bool(torch.isfinite(q_state).all().item()):
            raise RuntimeError("PSSF runtime Q state contains NaN/Inf.")
        if q_state.numel() and (
            float(q_state.min()) < 0.0 or float(q_state.max()) > 1.0
        ):
            raise RuntimeError("PSSF runtime Q state is outside [0,1].")
        self.q_state.copy_(q_state.to(dtype=self.dtype, device="cpu"))


class PSSFHistoryBank:
    MAP_NAMES = (
        "q_prev_37",
        "teacher_soft_37",
        "teacher_bin_37",
        "student_soft_37",
        "temporal_mean_37",
        "temporal_var_37",
    )

    def __init__(
        self,
        sample_count,
        patch_size=37,
        horizon=3,
        dtype=torch.float16,
    ):
        self.sample_count = int(sample_count)
        self.patch_size = int(patch_size)
        self.horizon = int(horizon)
        self.num_slots = self.horizon + 1
        self.dtype = dtype
        shape = (
            self.num_slots,
            self.sample_count,
            1,
            self.patch_size,
            self.patch_size,
        )
        self.maps = {
            name: torch.zeros(shape, dtype=dtype, device="cpu")
            for name in self.MAP_NAMES
        }
        self.epoch_tag = torch.full(
            (self.num_slots, self.sample_count),
            -1,
            dtype=torch.int32,
            device="cpu",
        )
        self.target_consumed = torch.zeros(
            (self.num_slots, self.sample_count),
            dtype=torch.bool,
            device="cpu",
        )
        self.segment_start_epoch = 1
        self._seen_epoch = None
        self._seen = None

    def _indices(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long, device="cpu")
        if indices.ndim != 1:
            raise RuntimeError("PSSF history indices must be [B].")
        if indices.numel() and (
            int(indices.min()) < 0 or int(indices.max()) >= self.sample_count
        ):
            raise RuntimeError("PSSF history index is out of range.")
        return indices

    def begin_epoch(self, epoch):
        self._seen_epoch = int(epoch)
        self._seen = torch.zeros(self.sample_count, dtype=torch.bool)

    def compute_temporal_stats(
        self,
        indices,
        epoch,
        current_teacher_soft_37,
        history_window=3,
    ):
        indices = self._indices(indices)
        epoch = int(epoch)
        values = []
        first_epoch = max(
            self.segment_start_epoch,
            epoch - int(history_window) + 1,
        )
        for historical_epoch in range(first_epoch, epoch):
            slot = historical_epoch % self.num_slots
            tags = self.epoch_tag[slot].index_select(0, indices)
            if not bool((tags == historical_epoch).all().item()):
                raise RuntimeError(
                    "PSSF temporal history tag mismatch: "
                    f"expected epoch {historical_epoch}."
                )
            values.append(
                self.maps["teacher_soft_37"][slot]
                .index_select(0, indices)
                .float()
            )
        values.append(current_teacher_soft_37.detach().cpu().float())
        stacked = torch.stack(values, dim=0)
        mean = stacked.mean(dim=0)
        variance = stacked.var(dim=0, correction=0)
        return mean, variance

    def write(
        self,
        indices,
        epoch,
        q_prev_37,
        teacher_soft_37,
        teacher_bin_37,
        student_soft_37,
        temporal_mean_37,
        temporal_var_37,
    ):
        indices = self._indices(indices)
        epoch = int(epoch)
        if self._seen_epoch != epoch or self._seen is None:
            raise RuntimeError("PSSF history begin_epoch() was not called.")
        if bool(self._seen.index_select(0, indices).any().item()):
            raise RuntimeError(
                f"PSSF sample appeared more than once in epoch {epoch}."
            )
        slot = epoch % self.num_slots
        existing_tags = self.epoch_tag[slot].index_select(0, indices)
        occupied = existing_tags >= self.segment_start_epoch
        if bool(occupied.any().item()):
            allowed_age = existing_tags <= epoch - self.num_slots
            consumed = self.target_consumed[slot].index_select(0, indices)
            if not bool((~occupied | (allowed_age & consumed)).all().item()):
                raise RuntimeError(
                    f"PSSF attempted to overwrite unconsumed history at epoch {epoch}."
                )

        values = {
            "q_prev_37": q_prev_37,
            "teacher_soft_37": teacher_soft_37,
            "teacher_bin_37": teacher_bin_37,
            "student_soft_37": student_soft_37,
            "temporal_mean_37": temporal_mean_37,
            "temporal_var_37": temporal_var_37,
        }
        expected = (len(indices), 1, self.patch_size, self.patch_size)
        for name, value in values.items():
            value = value.detach().cpu().float()
            if tuple(value.shape) != expected:
                raise RuntimeError(
                    f"PSSF history {name} shape mismatch: {list(value.shape)}."
                )
            if not bool(torch.isfinite(value).all().item()):
                raise RuntimeError(f"PSSF history {name} contains NaN/Inf.")
            self.maps[name][slot].index_copy_(
                0, indices, value.to(dtype=self.dtype)
            )
        self.epoch_tag[slot].index_fill_(0, indices, epoch)
        self.target_consumed[slot].index_fill_(0, indices, False)
        self._seen.index_fill_(0, indices, True)

    def matured_context(self, indices, current_epoch, device):
        indices = self._indices(indices)
        current_epoch = int(current_epoch)
        source_epoch = current_epoch - self.horizon
        if source_epoch < self.segment_start_epoch:
            return None
        source_slot = source_epoch % self.num_slots
        source_tags = self.epoch_tag[source_slot].index_select(0, indices)
        if not bool((source_tags == source_epoch).all().item()):
            raise RuntimeError(
                f"PSSF source history tag mismatch for epoch {source_epoch}."
            )
        output = {
            name: self.maps[name][source_slot]
            .index_select(0, indices)
            .to(device=device, dtype=torch.float32, non_blocking=True)
            for name in self.MAP_NAMES
        }
        future = []
        for future_epoch in range(source_epoch + 1, current_epoch + 1):
            slot = future_epoch % self.num_slots
            tags = self.epoch_tag[slot].index_select(0, indices)
            if not bool((tags == future_epoch).all().item()):
                raise RuntimeError(
                    f"PSSF future history tag mismatch for epoch {future_epoch}."
                )
            future.append(
                self.maps["teacher_bin_37"][slot]
                .index_select(0, indices)
                .to(device=device, dtype=torch.float32, non_blocking=True)
            )
        output["future_teacher_bin_37"] = torch.stack(future, dim=0)
        output["source_epoch"] = source_epoch
        return output

    def mark_consumed(self, indices, source_epoch):
        indices = self._indices(indices)
        slot = int(source_epoch) % self.num_slots
        tags = self.epoch_tag[slot].index_select(0, indices)
        if not bool((tags == int(source_epoch)).all().item()):
            raise RuntimeError("PSSF cannot consume a mismatched history slot.")
        self.target_consumed[slot].index_fill_(0, indices, True)

    def end_epoch(self, epoch):
        if self._seen_epoch != int(epoch) or self._seen is None:
            raise RuntimeError("PSSF history epoch bookkeeping is inactive.")
        if not bool(self._seen.all().item()):
            missing = torch.nonzero(~self._seen, as_tuple=False).flatten()[:20]
            raise RuntimeError(
                f"PSSF epoch {epoch} did not visit all samples; "
                f"first missing indices={missing.tolist()}."
            )
        self._seen_epoch = None
        self._seen = None

    def clear(self, next_epoch):
        for tensor in self.maps.values():
            tensor.zero_()
        self.epoch_tag.fill_(-1)
        self.target_consumed.zero_()
        self.segment_start_epoch = int(next_epoch)
        self._seen_epoch = None
        self._seen = None

    def state_dict(self):
        return {
            "schema_version": "pssf_history_bank_v1",
            "sample_count": self.sample_count,
            "patch_size": self.patch_size,
            "horizon": self.horizon,
            "num_slots": self.num_slots,
            "dtype": str(self.dtype).replace("torch.", ""),
            "segment_start_epoch": self.segment_start_epoch,
            "maps": self.maps,
            "epoch_tag": self.epoch_tag,
            "target_consumed": self.target_consumed,
        }

    def load_state_dict(self, state):
        if state.get("schema_version") != "pssf_history_bank_v1":
            raise RuntimeError("PSSF history bank schema mismatch.")
        expected = (
            self.sample_count,
            self.patch_size,
            self.horizon,
            self.num_slots,
        )
        actual = (
            int(state.get("sample_count", -1)),
            int(state.get("patch_size", -1)),
            int(state.get("horizon", -1)),
            int(state.get("num_slots", -1)),
        )
        if actual != expected:
            raise RuntimeError(
                f"PSSF history bank dimensions mismatch: {actual} != {expected}."
            )
        expected_dtype = str(self.dtype).replace("torch.", "")
        if str(state.get("dtype", "")) != expected_dtype:
            raise RuntimeError(
                "PSSF history bank dtype mismatch: "
                f"{state.get('dtype')!r} != {expected_dtype!r}."
            )
        maps = state.get("maps")
        if not isinstance(maps, dict) or set(maps) != set(self.MAP_NAMES):
            raise RuntimeError("PSSF runtime history map keys mismatch.")
        for name in self.MAP_NAMES:
            value = maps.get(name)
            if not torch.is_tensor(value) or tuple(value.shape) != tuple(
                self.maps[name].shape
            ):
                raise RuntimeError(f"PSSF runtime history map mismatch: {name}.")
            if not bool(torch.isfinite(value).all().item()):
                raise RuntimeError(f"PSSF runtime history map contains NaN/Inf: {name}.")
            if value.numel() and (
                float(value.min()) < 0.0 or float(value.max()) > 1.0
            ):
                raise RuntimeError(
                    f"PSSF runtime history map is outside [0,1]: {name}."
                )
            self.maps[name].copy_(value.to(dtype=self.dtype, device="cpu"))

        epoch_tag = state.get("epoch_tag")
        target_consumed = state.get("target_consumed")
        expected_tag_shape = (self.num_slots, self.sample_count)
        if not torch.is_tensor(epoch_tag) or tuple(epoch_tag.shape) != expected_tag_shape:
            raise RuntimeError("PSSF runtime history epoch tags mismatch.")
        if (
            not torch.is_tensor(target_consumed)
            or tuple(target_consumed.shape) != expected_tag_shape
        ):
            raise RuntimeError("PSSF runtime history consumed flags mismatch.")
        epoch_tag = epoch_tag.to(dtype=torch.int32, device="cpu")
        target_consumed = target_consumed.to(dtype=torch.bool, device="cpu")
        if bool((target_consumed & (epoch_tag < 0)).any().item()):
            raise RuntimeError(
                "PSSF runtime marks an empty history slot as consumed."
            )
        segment_start_epoch = int(state.get("segment_start_epoch", -1))
        if segment_start_epoch < 1:
            raise RuntimeError(
                "PSSF history segment_start_epoch must be positive."
            )

        self.epoch_tag.copy_(epoch_tag)
        self.target_consumed.copy_(
            target_consumed
        )
        self.segment_start_epoch = segment_start_epoch
        self._seen_epoch = None
        self._seen = None


def save_runtime_payload_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    digest = file_sha256(temporary)
    os.replace(temporary, path)
    return digest
