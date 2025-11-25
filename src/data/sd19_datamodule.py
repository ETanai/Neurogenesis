import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.error import URLError
from urllib.request import urlopen

from PIL import Image
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

DEFAULT_SD19_URL = "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"


class SD19ImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.loader(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label


class _ListSampler(Sampler[int]):
    """Sampler that draws from a changing list of indices."""

    def __init__(self, indices: Sequence[int]):
        self.set(indices)

    def set(self, indices: Sequence[int]) -> None:
        idxs = list(indices)
        self.indices = idxs
        # Maintain compatibility with helpers expecting `sampler.idxs`.
        self.idxs = idxs

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class SD19DataModule(pl.LightningDataModule):
    """
    LightningDataModule for NIST SD-19 “by_class” folder.

    New:
    - per-class / total sample limits for combined/class loaders & datasets
      -> get_combined_dataloader(..., per_class_limit=..., max_total=...)
      -> get_class_dataloader(..., limit=...)
      -> ... same for get_*_dataset(...)
    - cached class->indices for speed
    - deterministic shuffling with seed
    """

    def __init__(
        self,
        data_dir: str = "./data/SD19/by_class",
        batch_size: int = 128,
        num_workers: int = 4,
        classes: Sequence[int] | None = None,
        val_split: float = 0.2,
        image_size: int | None = None,
        # defaults for limiting (optional)
        default_per_class_limit_train: Optional[int] = None,
        default_per_class_limit_val: Optional[int] = None,
        index_seed: int = 42,
        download: bool = True,
        download_url: str | None = DEFAULT_SD19_URL,
        download_root: Optional[str] = None,
    ):
        if download_url is None:
            download_url = DEFAULT_SD19_URL
        if download_root is None:
            download_root = str(Path(data_dir).parent)
        super().__init__()
        self.save_hyperparameters()
        self.index_seed = index_seed
        self.class_to_indices: Dict[int, List[int]] = {}
        self._offline_resize_done = False

        # define transforms once (will be rebuilt in setup for image_size)
        self.train_tfms = transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor()])
        self.val_tfms = transforms.ToTensor()

    # ---------- utilities ----------

    def _rng(self, seed: Optional[int] = None) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(self.index_seed if seed is None else int(seed))
        return g

    def _build_class_index_cache(self):
        """Cache list of sample indices per class label (fast lookup later)."""
        self.class_to_indices.clear()
        for i, (_, lbl) in enumerate(self.full.imgs):
            self.class_to_indices.setdefault(lbl, []).append(i)

        # deterministic per-class shuffles (so selecting first N is reproducible)
        g = self._rng()
        for lbl, lst in self.class_to_indices.items():
            if len(lst) > 1:
                perm = torch.randperm(len(lst), generator=g).tolist()
                self.class_to_indices[lbl] = [lst[j] for j in perm]

    def _select_indices_for_classes(
        self,
        class_ids: Sequence[int],
        per_class_limit: Optional[int] = None,
        max_total: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[int]:
        """
        Select indices for given classes with optional per-class and total limits.
        - If per_class_limit is given: take up to N per class.
        - If max_total is given: cap overall size fairly (round-robin over classes).
        Deterministic given 'seed'.
        """
        if not class_ids:
            return []

        # start from cached per-class lists (already shuffled deterministically in setup)
        per_class_lists = []
        for c in class_ids:
            lst = self.class_to_indices.get(int(c), [])
            if per_class_limit is not None and per_class_limit > 0:
                lst = lst[:per_class_limit]
            per_class_lists.append(lst)

        if max_total is None or max_total <= 0:
            # simple concat
            out = [i for lst in per_class_lists for i in lst]
            return out

        # fair cap: round-robin until reaching max_total
        # re-shuffle order of classes deterministically if a seed is provided
        order = list(range(len(per_class_lists)))
        if len(order) > 1:
            g = self._rng(seed)
            perm = torch.randperm(len(order), generator=g).tolist()
            order = [order[j] for j in perm]

        cursors = [0] * len(per_class_lists)
        total = 0
        out: List[int] = []
        while total < max_total:
            progressed = False
            for k in order:
                lst = per_class_lists[k]
                cur = cursors[k]
                if cur < len(lst):
                    out.append(lst[cur])
                    cursors[k] += 1
                    total += 1
                    progressed = True
                    if total >= max_total:
                        break
            if not progressed:
                break  # all lists exhausted
        return out

    # ---------- Lightning hooks ----------

    def setup(self, stage=None):
        self._ensure_dataset_ready()
        # build transforms here so they see image_size
        self.train_tfms = self._make_tfms(train=True)
        self.val_tfms = self._make_tfms(train=False)

        samples, class_to_idx = self._build_or_load_index()

        self.full = SD19ImageFolder(self.hparams.data_dir, transform=None)
        self.full.samples = samples  # <— reuse (no rescan)
        self.full.imgs = samples  # alias used internally
        self.full.targets = [lbl for _, lbl in samples]
        self.full.class_to_idx = class_to_idx
        self._build_class_index_cache()

        targets = torch.tensor([lbl for _, lbl in self.full.samples], dtype=torch.long)

        N = len(self.full)
        perm = torch.randperm(N, generator=self._rng())
        n_val = int(self.hparams.val_split * N)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        if self.hparams.classes is not None:
            cls = torch.tensor(self.hparams.classes)
            mask = torch.isin(targets, cls)
            train_idx = train_idx[mask[train_idx]]
            val_idx = val_idx[mask[val_idx]]

        # apply optional global defaults (kept None by default to preserve old behavior)
        if self.hparams.default_per_class_limit_train:
            # limit by class among currently selected train_idx
            selected = []
            per_class_count = {}
            # build reverse: index -> label
            for i in train_idx.tolist():
                lbl = self.full.imgs[i][1]
                cnt = per_class_count.get(lbl, 0)
                if cnt < int(self.hparams.default_per_class_limit_train):
                    selected.append(i)
                    per_class_count[lbl] = cnt + 1
            train_idx = torch.tensor(selected, dtype=torch.long)

        if self.hparams.default_per_class_limit_val:
            selected = []
            per_class_count = {}
            for i in val_idx.tolist():
                lbl = self.full.imgs[i][1]
                cnt = per_class_count.get(lbl, 0)
                if cnt < int(self.hparams.default_per_class_limit_val):
                    selected.append(i)
                    per_class_count[lbl] = cnt + 1
            val_idx = torch.tensor(selected, dtype=torch.long)

        self.train_sampler = _ListSampler(train_idx.tolist())
        self.val_sampler = _ListSampler(val_idx.tolist())

        # datasets WITH transforms
        train_ds = _TransformedView(self.full, self.train_tfms)
        val_ds = _TransformedView(self.full, self.val_tfms)

        self._train_loader = DataLoader(
            train_ds,
            batch_size=self.hparams.batch_size,
            sampler=self.train_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )
        self._val_loader = DataLoader(
            val_ds,
            batch_size=self.hparams.batch_size,
            sampler=self.val_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        return self._val_loader

    # ---------- NEW: limited class/combined loaders ----------

    def get_class_dataloader(
        self,
        class_id: int,
        limit: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> DataLoader:
        """Switch train loader to only that class (optionally limit samples)."""
        full_list = self.class_to_indices.get(int(class_id), [])
        if limit is not None and limit > 0:
            # deterministic subselect
            g = self._rng(seed)
            if len(full_list) > 1:
                perm = torch.randperm(len(full_list), generator=g).tolist()
                full_list = [full_list[j] for j in perm]
            idxs = full_list[:limit]
        else:
            idxs = full_list
        self.train_sampler.set(idxs)
        return self._train_loader

    def get_combined_dataloader(
        self,
        class_ids: Sequence[int],
        per_class_limit: Optional[int] = None,
        max_total: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> DataLoader:
        """Switch train loader to multiple classes with optional caps."""
        idxs = self._select_indices_for_classes(
            class_ids, per_class_limit=per_class_limit, max_total=max_total, seed=seed
        )
        self.train_sampler.set(idxs)
        return self._train_loader

    # ---------- NEW: limited datasets ----------

    def get_class_dataset(
        self,
        class_id: int,
        limit: Optional[int] = None,
        seed: Optional[int] = None,
        use_val_transforms: bool = True,
    ) -> Subset:
        """Return a Subset for a single class (optionally limited)."""
        lst = self.class_to_indices.get(int(class_id), [])
        if limit is not None and limit > 0:
            g = self._rng(seed)
            if len(lst) > 1:
                perm = torch.randperm(len(lst), generator=g).tolist()
                lst = [lst[j] for j in perm]
            lst = lst[:limit]
        transform = self.val_tfms if use_val_transforms else self.train_tfms
        ds = SD19ImageFolder(self.hparams.data_dir, transform=transform)
        return Subset(ds, lst)

    def get_combined_dataset(
        self,
        class_ids: Sequence[int],
        per_class_limit: Optional[int] = None,
        max_total: Optional[int] = None,
        seed: Optional[int] = None,
        use_val_transforms: bool = True,
    ) -> Subset:
        """Return a Subset for multiple classes (optionally limited)."""
        idxs = self._select_indices_for_classes(
            class_ids, per_class_limit=per_class_limit, max_total=max_total, seed=seed
        )
        transform = self.val_tfms if use_val_transforms else self.train_tfms
        ds = SD19ImageFolder(self.hparams.data_dir, transform=transform)
        return Subset(ds, idxs)

    # ---------- transforms ----------

    def _make_tfms(self, train: bool):
        tfms = []
        if self.hparams.image_size is not None and not self._offline_resize_done:
            tfms.append(transforms.Resize((self.hparams.image_size, self.hparams.image_size)))
        if train:
            tfms += [transforms.RandomRotation(10), transforms.ToTensor()]
        else:
            tfms.append(transforms.ToTensor())
        return transforms.Compose(tfms)

    def _cache_path(self) -> Path:
        root = Path(self.hparams.data_dir)
        return root / ".sd19_index_cache.json"

    def _build_or_load_index(self):
        """Build or load cached (samples, class_to_idx)."""
        cache_file = self._cache_path()
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                samples = [(str(p), int(l)) for p, l in data["samples"]]
                class_to_idx = {k: int(v) for k, v in data["class_to_idx"].items()}
                return samples, class_to_idx
            except Exception:
                pass  # fall through to rebuild

        # Fallback: build via ImageFolder once
        tmp = SD19ImageFolder(self.hparams.data_dir, transform=None)
        samples = [(p, int(l)) for p, l in tmp.samples]
        class_to_idx = {k: int(v) for k, v in tmp.class_to_idx.items()}

        # save cache (small ~ tens of MB depending on dataset)
        try:
            payload = {
                "samples": samples,
                "class_to_idx": class_to_idx,
            }
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass
        return samples, class_to_idx

    # ---------- download helpers ----------

    def _ensure_dataset_ready(self) -> None:
        root = Path(self.hparams.data_dir)
        if root.exists() and any(root.iterdir()):
            self._prepare_image_resolution(root)
            return

        if not bool(self.hparams.download):
            raise FileNotFoundError(
                f"SD-19 data directory '{root}' is missing and automatic download is disabled."
            )

        url = self.hparams.download_url or DEFAULT_SD19_URL
        target_root = Path(self.hparams.download_root)
        target_root.mkdir(parents=True, exist_ok=True)
        archive_path = target_root / Path(url).name

        if not archive_path.exists():
            self._download_archive(url, archive_path)
        self._extract_archive(archive_path, target_root)

        if not root.exists() or not any(root.iterdir()):
            raise FileNotFoundError(
                f"Failed to locate SD-19 data at '{root}' after extracting {archive_path}."
            )
        self._prepare_image_resolution(root)

    def _download_archive(self, url: str, dest: Path) -> None:
        try:
            with urlopen(url) as response, open(dest, "wb") as fh:
                shutil.copyfileobj(response, fh)
        except URLError as err:  # pragma: no cover - network errors hard to unit test
            raise RuntimeError(f"Unable to download SD-19 archive from {url}: {err}") from err

    def _extract_archive(self, archive_path: Path, target_root: Path) -> None:
        try:
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(target_root)
        except zipfile.BadZipFile as err:
            raise RuntimeError(
                f"SD-19 archive '{archive_path}' is corrupted or not a valid zip file."
            ) from err

    def _prepare_image_resolution(self, root: Path) -> None:
        size = self.hparams.image_size
        if size is None:
            return
        marker = root / f".resized_{size}.done"
        if marker.exists():
            self._offline_resize_done = True
            return
        if not root.exists():
            raise FileNotFoundError(f"Cannot resize SD-19 images; '{root}' does not exist.")

        supported_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in supported_suffixes:
                continue
            self._resize_image_file(path, size)
        marker.write_text("ok", encoding="utf-8")
        self._offline_resize_done = True

    def _resize_image_file(self, path: Path, size: int) -> None:
        with Image.open(path) as img:
            img = img.convert("L")
            if img.size != (size, size):
                img = img.resize((size, size), Image.LANCZOS)
            img.save(path)


class _TransformedView(torch.utils.data.Dataset):
    """Shares the same samples list; only changes the transform used at __getitem__ time."""

    def __init__(self, base: SD19ImageFolder, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base.samples)

    def __getitem__(self, index):
        path, label = self.base.samples[index]
        img = self.base.loader(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label
