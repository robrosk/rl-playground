import csv

from rl_playground.core.callbacks import CSVLogger
from rl_playground.losses.composite import CompositeLoss


class Term:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __call__(self, inputs):
        return self.value * inputs["scale"], {"seen": inputs["scale"]}


def test_composite_is_the_weighted_sum_with_metrics_per_term():
    loss = CompositeLoss([(1.0, Term("policy", 2.0)), (0.5, Term("value", 4.0))])
    total, metrics = loss({"scale": 3.0})
    assert total == 1.0 * 6.0 + 0.5 * 12.0
    assert metrics["loss/policy"] == 6.0 and metrics["loss/value"] == 12.0
    assert metrics["loss/total"] == 12.0
    assert metrics["policy/seen"] == 3.0


def test_csv_logger_writes_a_header_then_rows(tmp_path):
    path = tmp_path / "log.csv"
    logger = CSVLogger(path)
    logger.on_update(1, {"loss": 0.5})
    logger.on_update(2, {"loss": 0.25})
    rows = list(csv.reader(path.open()))
    assert rows == [["step", "loss"], ["1", "0.5"], ["2", "0.25"]]
