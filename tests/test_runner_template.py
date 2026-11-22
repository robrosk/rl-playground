from rl_playground.core.callbacks import Callback, ContractionMonitor
from rl_playground.runners.base import Runner
from rl_playground.runners.planning import PlanningRunner


class Recorder(Callback):
    def __init__(self):
        self.events = []

    def on_run_start(self, runner):
        self.events.append("start")

    def on_update(self, step, metrics):
        self.events.append(("update", step, metrics["n"]))

    def on_run_end(self, runner):
        self.events.append("end")


class EveryThird(Runner):
    def __init__(self, callbacks=()):
        super().__init__(callbacks)
        self.cleared = 0

    def step(self):
        pass

    def ready_to_update(self):
        return self.t % 3 == 0

    def update(self):
        return {"n": self.updates}

    def after_update(self, metrics):
        self.cleared += 1


def test_run_calls_hooks_in_order():
    rec = Recorder()
    runner = EveryThird([rec]).run(budget=7)
    assert runner.t == 7
    assert runner.updates == 2 and runner.cleared == 2
    assert rec.events == ["start", ("update", 3, 0), ("update", 6, 1), "end"]


class FakePlanner:
    def __init__(self, deltas):
        self.deltas = iter(deltas)

    def sweep(self):
        return next(self.deltas)


def test_planning_runner_stops_below_tolerance():
    runner = PlanningRunner(FakePlanner([1.0, 0.5, 1e-9, 7.0]), tol=1e-6).run(budget=100)
    assert runner.t == 3


def test_contraction_monitor_flags_a_sweep_that_grows_too_fast():
    monitor = ContractionMonitor(gamma=0.9)
    PlanningRunner(FakePlanner([1.0, 0.9, 0.85, 0.1, 1e-12]), tol=1e-9, callbacks=[monitor]).run(100)
    assert monitor.deltas[:4] == [1.0, 0.9, 0.85, 0.1]
    assert [(after) for _, _, after in monitor.violations] == [0.85]  # 0.85 > 0.9 * 0.9
