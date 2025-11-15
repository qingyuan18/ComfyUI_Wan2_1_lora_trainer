# Import musubi_utils first to ensure it's available for other modules
from . import musubi_utils
# Expose musubi_utils as a top-level module to support loaders that break package context
import sys as _sys
_sys.modules.setdefault("musubi_utils", musubi_utils)

# Pre-import key sibling modules under package context and alias them for loaders that break package context
try:
    from . import wan_train_network as _wan_train_network
    _sys.modules.setdefault("wan_train_network", _wan_train_network)
except Exception:
    pass
try:
    from . import hv_train_network as _hv_train_network
    _sys.modules.setdefault("hv_train_network", _hv_train_network)
except Exception:
    pass
try:
    from . import wan_cache_latents as _wan_cache_latents
    _sys.modules.setdefault("wan_cache_latents", _wan_cache_latents)
except Exception:
    pass
try:
    from . import cache_latents as _cache_latents
    _sys.modules.setdefault("cache_latents", _cache_latents)
except Exception:
    pass
try:
    from . import wan_cache_text_encoder_outputs as _wan_cache_text_encoder_outputs
    _sys.modules.setdefault("wan_cache_text_encoder_outputs", _wan_cache_text_encoder_outputs)
except Exception:
    pass
try:
    from . import cache_text_encoder_outputs as _cache_text_encoder_outputs
    _sys.modules.setdefault("cache_text_encoder_outputs", _cache_text_encoder_outputs)
except Exception:
    pass
# Also expose subpackages used in fallbacks
try:
    from . import dataset as _dataset
    _sys.modules.setdefault("dataset", _dataset)
except Exception:
    pass
try:
    from . import wan as _wan_pkg
    _sys.modules.setdefault("wan", _wan_pkg)
except Exception:
    pass


# Importar los mapeos de nodos del archivo wan_trainer_nodes.py
# Usamos alias para evitar conflictos de nombres si ambos archivos tuvieran las mismas variables
from .wan_trainer_nodes import NODE_CLASS_MAPPINGS as TRAINER_NODE_CLASS_MAPPINGS, \
                               NODE_DISPLAY_NAME_MAPPINGS as TRAINER_NODE_DISPLAY_NAME_MAPPINGS

# Importar los mapeos de nodos del nuevo archivo wan_complements.py
from .wan_complements import NODE_CLASS_MAPPINGS as COMPLEMENTS_NODE_CLASS_MAPPINGS, \
                               NODE_DISPLAY_NAME_MAPPINGS as COMPLEMENTS_NODE_DISPLAY_NAME_MAPPINGS

# Combinar los diccionarios de mapeo de clases de nodos
NODE_CLASS_MAPPINGS = {**TRAINER_NODE_CLASS_MAPPINGS, **COMPLEMENTS_NODE_CLASS_MAPPINGS}

# Combinar los diccionarios de mapeo de nombres de visualizaci�n de nodos
NODE_DISPLAY_NAME_MAPPINGS = {**TRAINER_NODE_DISPLAY_NAME_MAPPINGS, **COMPLEMENTS_NODE_DISPLAY_NAME_MAPPINGS}

# Exportar las variables combinadas para que ComfyUI las descubra
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
MAGENTA = "\033[35m"
RESET = "\033[0m"
print(f"{MAGENTA}\033[4mMusubiTuner] WAN 2.1 LORA TRAINER.{RESET}") # Magenta y subrayado
