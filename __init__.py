# Import musubi_utils first to ensure it's available for other modules
from . import musubi_utils
# Expose musubi_utils as a top-level module to support loaders that break package context
import sys as _sys
_sys.modules.setdefault("musubi_utils", musubi_utils)

# First, expose subpackages used in fallbacks - BEFORE importing modules that depend on them
try:
    from . import dataset as _dataset
    _sys.modules.setdefault("dataset", _dataset)
except Exception as e:
    print(f"Warning: Could not import dataset: {e}")
    pass
try:
    from . import wan as _wan_pkg
    _sys.modules.setdefault("wan", _wan_pkg)
except Exception as e:
    print(f"Warning: Could not import wan: {e}")
    pass
try:
    from . import modules as _modules
    _sys.modules.setdefault("modules", _modules)
except Exception as e:
    print(f"Warning: Could not import modules: {e}")
    pass
try:
    from . import networks as _networks
    _sys.modules.setdefault("networks", _networks)
except Exception as e:
    print(f"Warning: Could not import networks: {e}")
    pass
try:
    from . import hunyuan_model as _hunyuan_model
    _sys.modules.setdefault("hunyuan_model", _hunyuan_model)
except Exception as e:
    print(f"Warning: Could not import hunyuan_model: {e}")
    pass
try:
    from . import train_utils as _train_utils
    _sys.modules.setdefault("train_utils", _train_utils)
except Exception as e:
    print(f"Warning: Could not import train_utils: {e}")
    pass

# Now import key sibling modules under package context and alias them for loaders that break package context
try:
    from . import hv_generate_video as _hv_generate_video
    _sys.modules.setdefault("hv_generate_video", _hv_generate_video)
except Exception as e:
    print(f"Warning: Could not import hv_generate_video: {e}")
    pass
try:
    from . import wan_generate_video as _wan_generate_video
    _sys.modules.setdefault("wan_generate_video", _wan_generate_video)
except Exception as e:
    print(f"Warning: Could not import wan_generate_video: {e}")
    pass
try:
    from . import hv_train_network as _hv_train_network
    _sys.modules.setdefault("hv_train_network", _hv_train_network)
except Exception as e:
    print(f"Warning: Could not import hv_train_network: {e}")
    pass
try:
    from . import wan_train_network as _wan_train_network
    _sys.modules.setdefault("wan_train_network", _wan_train_network)
except Exception as e:
    print(f"Warning: Could not import wan_train_network: {e}")
    pass
try:
    from . import cache_latents as _cache_latents
    _sys.modules.setdefault("cache_latents", _cache_latents)
except Exception as e:
    print(f"Warning: Could not import cache_latents: {e}")
    pass
try:
    from . import wan_cache_latents as _wan_cache_latents
    _sys.modules.setdefault("wan_cache_latents", _wan_cache_latents)
except Exception as e:
    print(f"Warning: Could not import wan_cache_latents: {e}")
    pass
try:
    from . import cache_text_encoder_outputs as _cache_text_encoder_outputs
    _sys.modules.setdefault("cache_text_encoder_outputs", _cache_text_encoder_outputs)
except Exception as e:
    print(f"Warning: Could not import cache_text_encoder_outputs: {e}")
    pass
try:
    from . import wan_cache_text_encoder_outputs as _wan_cache_text_encoder_outputs
    _sys.modules.setdefault("wan_cache_text_encoder_outputs", _wan_cache_text_encoder_outputs)
except Exception as e:
    print(f"Warning: Could not import wan_cache_text_encoder_outputs: {e}")
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
