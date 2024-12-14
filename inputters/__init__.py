from .comet import Inputter as comet
from .comet_without_situation import Inputter as comet_without_situation
from .comet_without_self_effect import Inputter as comet_without_self_effect
from .comet_without_other_effect import Inputter as comet_without_other_effect
from .comet_without_cause import Inputter as comet_without_cause
from .comet_without_executor import Inputter as comet_without_executor
from .strat import Inputter as strat
from .vanilla import Inputter as vanilla
from .causal import Inputter as causal

inputters = {
    'vanilla': vanilla,
    'strat': strat,
    'comet': comet,
    'comet_without_situation': comet_without_situation,
    'comet_without_self_effect': comet_without_self_effect,
    'comet_without_other_effect': comet_without_other_effect,
    'comet_without_cause': comet_without_cause,
    'comet_without_executor': comet_without_executor,
    'causal': causal
}



