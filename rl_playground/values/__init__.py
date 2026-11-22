"""How value is stored (Bridge pattern).

Algorithms talk to ``VFunction`` and ``QFunction``. ``TabularV``/``TabularQ`` and
``NeuralV``/``NeuralQ`` implement them, so Q-learning runs on a table or a network unchanged.
The neural ones live in ``values/neural.py`` and are imported from there, so the tabular half
of the library works without torch.
"""

from rl_playground.values.base import QFunction, VFunction
from rl_playground.values.tabular import TabularQ, TabularV

__all__ = ["QFunction", "TabularQ", "TabularV", "VFunction"]
