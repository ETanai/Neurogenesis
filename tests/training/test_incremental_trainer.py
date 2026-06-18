import torch

from training.incremental_trainer import IncrementalTrainer


class DummyAE:
    def parameters(self):
        return iter([])


class ReplayStore:
    def __init__(self, classes):
        self._classes = list(classes)
        self.calls = []

    def available_classes(self):
        return self._classes

    def sample_images(self, class_id, replay_size):
        self.calls.append((int(class_id), int(replay_size)))
        return torch.full((int(replay_size), 4), float(class_id))


def test_incremental_paper_replay_samples_each_old_class():
    replay = ReplayStore(classes=[1, 7])
    trainer = IncrementalTrainer(
        ae=DummyAE(),
        ir=replay,
        base_lr=1.0e-3,
        epochs=1,
        replay_ratio=0.0,
        replay_mode="paper",
        replay_per_class_ratio=0.5,
        device=torch.device("cpu"),
    )
    inputs = torch.zeros(4, 1, 2, 2)

    mixed = trainer._augment_with_replay(inputs)

    assert mixed.shape == (8, 1, 2, 2)
    assert replay.calls == [(1, 2), (7, 2)]

