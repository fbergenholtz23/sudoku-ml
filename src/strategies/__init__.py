from .naked_single import naked_single
from .hidden_single import hidden_single
from .naked_pair import naked_pair
from .hidden_pair import hidden_pair
from .pointing_pairs import pointing_pairs

# Ordered by strategy complexity — simpler strategies first
STRATEGIES = [
    naked_single,
    hidden_single,
    pointing_pairs,
    naked_pair,
    hidden_pair,
]
