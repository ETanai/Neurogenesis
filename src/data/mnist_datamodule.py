from typing import Any, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torchvision import datasets, transforms


class _MemMNIST(Dataset):
    """MNIST fully in RAM, raw tensors in [0,1]. Uses external sampler for subsetting."""

    def __init__(self, train: bool, data_dir: str):
        t = transforms.ToTensor()
        ds = datasets.MNIST(data_dir, train=train, download=True, transform=t)
        self.data = ds.data.float().div(255).unsqueeze(1)  # [N,1,28,28]
        self.targets = ds.targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.targets[i]


class _ListSampler(Sampler[int]):
    """Sampler with mutable index list."""

    def __init__(self, idxs: Sequence[int]):
        self.idxs = list(idxs)

    def set(self, idxs: Sequence[int] | torch.Tensor) -> None:
        self.idxs = list(idxs)

    def __iter__(self):
        return iter(self.idxs)

    def __len__(self) -> int:
        return len(self.idxs)


class MNISTDataModule(pl.LightningDataModule):
    """
    LightningDataModule for MNIST.
    Fully in-RAM, single persistent loader, dynamic subset switching.

    Maintains original signatures:
      - train_dataloader()
      - val_dataloader()
      - get_class_dataloader(class_id)
      - get_combined_dataloader(class_ids)
    """

    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 4,
        data_dir: str = "./data",
        classes: Sequence[int] | None = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        # preserve signature params
        self.save_hyperparameters()

    def setup(self, stage: Any = None) -> None:
        # load full datasets into RAM
        self.full_train = _MemMNIST(train=True, data_dir=self.hparams.data_dir)
        self.full_val = _MemMNIST(train=False, data_dir=self.hparams.data_dir)

        # dynamic samplers over full sets
        train_idxs = list(range(len(self.full_train)))
        val_idxs = list(range(len(self.full_val)))

        # optional global class filter
        if self.hparams.classes is not None:
            cls = torch.as_tensor(self.hparams.classes)
            mask = torch.isin(self.full_train.targets, cls)
            train_idxs = mask.nonzero(as_tuple=True)[0].tolist()
            mask = torch.isin(self.full_val.targets, cls)
            val_idxs = mask.nonzero(as_tuple=True)[0].tolist()

        self.train_sampler = _ListSampler(train_idxs)
        self.val_sampler = _ListSampler(val_idxs)

        # persistent single loaders
        self._train_loader = DataLoader(
            self.full_train,
            batch_size=self.hparams.batch_size,
            sampler=self.train_sampler,
            num_workers=0,
            pin_memory=True,
        )
        self._val_loader = DataLoader(
            self.full_val,
            batch_size=self.hparams.batch_size,
            sampler=self.val_sampler,
            num_workers=0,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        return self._val_loader

    def get_class_dataloader(self, class_id: int) -> DataLoader:
        """Switch to only that digit class."""
        idxs = (self.full_train.targets == class_id).nonzero(as_tuple=True)[0]
        self.train_sampler.set(idxs)
        return self._train_loader

    def get_combined_dataloader(self, class_ids: Sequence[int]) -> DataLoader:
        """Switch to multiple classes."""
        cls = torch.as_tensor(list(class_ids))
        mask = torch.isin(self.full_train.targets, cls)
        idxs = mask.nonzero(as_tuple=True)[0]
        self.train_sampler.set(idxs)
        return self._train_loader

    def get_class_dataset(self, class_id: int, *, split: str = "train") -> Subset:
        ds = self.full_train if split == "train" else self.full_val
        t = ds.targets
        idxs = (t == class_id).nonzero(as_tuple=True)[0].tolist()
        return Subset(ds, idxs)

    def get_combined_dataset(self, class_ids: Sequence[int], *, split: str = "train") -> Subset:
        ds = self.full_train if split == "train" else self.full_val
        cls = torch.as_tensor(list(class_ids))
        t = ds.targets
        mask = torch.isin(t, cls)
        idxs = mask.nonzero(as_tuple=True)[0].tolist()
        return Subset(ds, idxs)

    def make_class_balanced_batch(
        self,
        classes: Sequence[int],
        samples_per_class: int,
        *,
        split: str = "train",  # "train" or "val"
        device: torch.device | None = None,  # defaults to self.hparams.device
        shuffle: bool = True,
        seed: int | None = None,
        allow_replacement: bool = False,
        return_labels: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, list[int]]]:
        """
        Build a batch with equal #samples per class, ordered by the 'classes' list.

        Returns:
            x:  Tensor [B, C, H, W] on the chosen device
            y:  (optional) LongTensor [B] of labels on same device
            splits: cumulative column counts after each class block
                    (useful for plot_partial_recon_grid_mlflow col_group_splits)

        Example:
            x, y, splits = dm.make_class_balanced_batch([4,6,8], 8)
            fig = plot_partial_recon_grid_mlflow(
                ae, x, view_shape=(1,28,28),
                ncols=x.size(0),
                col_group_titles=[str(c) for c in [4,6,8]],
                col_group_splits=splits,
            )
        """
        assert split in {"train", "val"}
        ds = self.full_train if split == "train" else self.full_val
        targets = ds.targets
        dev = device if device is not None else self.hparams.device

        # local RNG for reproducibility (optional)
        g = None
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(int(seed))

        imgs: list[torch.Tensor] = []
        labels: list[int] = []
        splits: list[int] = []
        total = 0

        # lazy import in case PIL images are returned
        try:
            from torchvision.transforms.functional import to_tensor as _to_tensor
        except Exception:
            _to_tensor = None  # dataset is expected to return tensors

        for c in classes:
            idxs = (targets == int(c)).nonzero(as_tuple=True)[0]
            n_avail = idxs.numel()
            if n_avail == 0:
                raise ValueError(f"No samples found for class {c} in split='{split}'.")

            if samples_per_class > n_avail and not allow_replacement:
                raise ValueError(
                    f"Requested {samples_per_class} samples for class {c}, "
                    f"but only {n_avail} available. "
                    f"Set allow_replacement=True to sample with replacement."
                )

            if samples_per_class <= n_avail:
                if shuffle:
                    perm = (
                        torch.randperm(n_avail, generator=g)
                        if g is not None
                        else torch.randperm(n_avail)
                    )
                    chosen = idxs[perm[:samples_per_class]]
                else:
                    chosen = idxs[:samples_per_class]
            else:
                # sample with replacement
                draw = torch.randint(0, n_avail, (samples_per_class,), generator=g)
                chosen = idxs[draw]

            for idx in chosen.tolist():
                x_i, y_i = ds[idx]
                # Convert to tensor if needed
                if not isinstance(x_i, torch.Tensor):
                    if _to_tensor is None:
                        raise TypeError(
                            "Dataset returned a non-tensor image and torchvision is unavailable."
                        )
                    x_i = _to_tensor(x_i)
                x_i = x_i.float()
                # Ensure channel dim exists: [C,H,W]
                if x_i.ndim == 2:
                    x_i = x_i.unsqueeze(0)
                # Normalize 0..255 -> 0..1 if needed
                if x_i.max() > 1.0:
                    x_i = x_i / 255.0

                imgs.append(x_i)
                labels.append(int(y_i))

            total += samples_per_class
            splits.append(total)

        x_batch = torch.stack(imgs, dim=0).to(dev, non_blocking=True)
        if return_labels:
            y_batch = torch.as_tensor(labels, dtype=torch.long, device=dev)
            return x_batch, y_batch, splits
        return x_batch

    def _add_gap_column(
        self, x: torch.Tensor, gap_after: int, gap_value: float = 1.0
    ) -> torch.Tensor:
        """
        Insert a single blank column (all gap_value) after `gap_after` images.
        Keeps shape [B, C, H, W]. Returns expanded tensor.
        """
        if gap_after <= 0 or gap_after >= x.size(0):
            return x
        C, H, W = x.shape[1:]
        gap = torch.full((1, C, H, W), gap_value, device=x.device, dtype=x.dtype)
        return torch.cat([x[:gap_after], gap, x[gap_after:]], dim=0)

    def make_grouped_batch_for_partial_plot(
        self,
        pretrained_classes: Sequence[int],
        novel_classes: Sequence[int],
        samples_per_class: int,
        *,
        split: str = "train",
        device: torch.device | None = None,
        shuffle: bool = True,
        seed: int | None = None,
        add_gap: bool = True,  # insert a visible blank column between groups
        gap_value: float = 1.0,  # 1.0 -> white gap for MNIST in [0,1]
    ):
        """
        Build a batch ordered as [pretrained..., novel...], plus group titles/splits
        for plot_partial_recon_grid_mlflow(...).

        Returns:
            x: Tensor [B or B+1, C, H, W]
            y: LongTensor [B] (labels; note: no label for the gap column)
            titles: ["Pre-trained", "Novel"]
            splits: [end_of_pretrained, total_cols]  # group-level cumulative ends
        """
        # use existing balanced-batch method to keep behavior consistent
        classes = list(pretrained_classes) + list(novel_classes)
        out = self.make_class_balanced_batch(
            classes,
            samples_per_class,
            split=split,
            device=device,
            shuffle=shuffle,
            seed=seed,
            allow_replacement=False,
            return_labels=True,
        )
        x, y, _ = out  # ignore per-class splits

        # compute group-level split
        n_pre_cols = samples_per_class * len(pretrained_classes)
        titles = ["Pre-trained", "Novel"]

        if add_gap:
            # insert a blank column to visually separate groups
            x = self._add_gap_column(x, n_pre_cols, gap_value=gap_value)
            # we don't add a label for the gap column; plotting doesn't use labels
            splits = [n_pre_cols, x.size(0)]
        else:
            splits = [n_pre_cols, x.size(0)]

        return x, y, titles, splits
