import pytest

from rl_playground.algorithms import ALGORITHMS
from rl_playground.core.registry import Registry


def test_add_get_and_make():
    reg = Registry("thing")
    reg.add("list", list)
    assert reg.get("list") is list
    assert reg.make("list", (1, 2)) == [1, 2]


def test_decorator_registers():
    reg = Registry("thing")

    @reg.register("double")
    def double(x):
        return 2 * x

    assert reg.make("double", 3) == 6


def test_lazy_entries_resolve_on_first_get():
    reg = Registry("thing")
    reg.add("sqrt", "math:sqrt")
    assert reg.make("sqrt", 9.0) == 3.0


def test_unknown_name_lists_known_names():
    reg = Registry("thing")
    reg.add("a", 1)
    with pytest.raises(KeyError, match="known: a"):
        reg.get("b")


def test_duplicate_names_are_refused():
    reg = Registry("thing")
    reg.add("a", 1)
    with pytest.raises(KeyError):
        reg.add("a", 2)


def test_every_algorithm_has_a_config():
    from rl_playground.core.config import CONFIGS

    assert set(ALGORITHMS.names()) == set(CONFIGS)
