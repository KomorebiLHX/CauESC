
from .strat_blenderbot_small import Model as strat_blenderbot_small
from .vanilla_blenderbot_small import Model as vanilla_blenderbot_small
from .comet_blenderbot_small import Model as comet_blenderbot_small
from .comet_blenderbot_small_single import Model as comet_blenderbot_small_single
from .comet_blenderbot_small_without_cause import Model as comet_blenderbot_small_without_cause
from .comet_blenderbot_small_without_situation import Model as comet_blenderbot_small_without_situation
from .comet_blenderbot_small_without_self_effect import Model as comet_blenderbot_small_without_self_effect
from .comet_blenderbot_small_without_other_effect import Model as comet_blenderbot_small_without_other_effect
from .comet_blenderbot_small_without_executor import Model as comet_blenderbot_small_without_executor
from .comet_blenderbot_small_with_multi_factor_decoder import Model as comet_blenderbot_small_with_multi_factor_decoder
from .comet_blenderbot_small_with_multi_factor_decoder_single import Model as comet_blenderbot_small_with_multi_factor_decoder_single
from .causal_blenderbot_small import Model as causal_blenderbot_small

from .strat_dialogpt import Model as strat_dialogpt
from .vanilla_dialogpt import Model as vanilla_dialogpt

models = {
    'vanilla_blenderbot_small': vanilla_blenderbot_small,
    'strat_blenderbot_small': strat_blenderbot_small,
    'comet_blenderbot_small': comet_blenderbot_small,
    'comet_blenderbot_small_single': comet_blenderbot_small_single,
    'comet_blenderbot_small_without_cause': comet_blenderbot_small_without_cause,
    'comet_blenderbot_small_without_situation': comet_blenderbot_small_without_situation,
    'comet_blenderbot_small_without_self_effect': comet_blenderbot_small_without_self_effect,
    'comet_blenderbot_small_without_other_effect': comet_blenderbot_small_without_other_effect,
    'comet_blenderbot_small_without_executor': comet_blenderbot_small_without_executor,
    'comet_blenderbot_small_with_multi_factor_decoder': comet_blenderbot_small_with_multi_factor_decoder,
    'comet_blenderbot_small_with_multi_factor_decoder_single': comet_blenderbot_small_with_multi_factor_decoder_single,
    'causal_blenderbot_small': causal_blenderbot_small,
    
    'vanilla_dialogpt': vanilla_dialogpt,
    'strat_dialogpt': strat_dialogpt,
}