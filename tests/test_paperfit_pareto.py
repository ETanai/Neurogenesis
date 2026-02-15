from scripts.run_paper_config import _pareto_ranks


def test_pareto_ranks_two_fronts() -> None:
    points = {
        "a": (0.2, 0.2),
        "b": (0.1, 0.3),
        "c": (0.3, 0.1),
        "d": (0.4, 0.4),
    }
    ranks = _pareto_ranks(points)
    assert ranks["a"] == 1
    assert ranks["b"] == 1
    assert ranks["c"] == 1
    assert ranks["d"] == 2
