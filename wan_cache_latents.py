# wan_cache_latents.py

import argparse
import os
# import glob # No parece usarse directamente, pero podr�a ser usado por cache_latents.encode_datasets
from typing import Optional, Union # Union no se usa, pero Optional s�
# import argparse # Ya importado arriba
import numpy as np
import torch
import logging

# Try relative imports first, fall back to absolute imports
try:
    from .dataset import config_utils as cfgutils
    from .dataset.config_utils import BlueprintGenerator, ConfigSanitizer
    from .dataset.image_video_dataset import ItemInfo, save_latent_cache_wan, ARCHITECTURE_WAN
    from .train_utils.model_utils import str_to_dtype
    from .wan.configs import wan_i2v_14B
    from .wan.modules.vae import WanVAE
    from .wan.modules.clip import CLIPModel
    from . import cache_latents as base_cache_latents_script
except ImportError:
    from dataset import config_utils as cfgutils
    from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
    from dataset.image_video_dataset import ItemInfo, save_latent_cache_wan, ARCHITECTURE_WAN
    from train_utils.model_utils import str_to_dtype
    from wan.configs import wan_i2v_14B
    from wan.modules.vae import WanVAE
    from wan.modules.clip import CLIPModel
    import cache_latents as base_cache_latents_script

logger = logging.getLogger(__name__)
# Configurar logging una sola vez, preferiblemente en __init__.py de tu paquete
if not logger.hasHandlers(): # Evitar m�ltiples handlers si el script se recarga
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s %(name)s] %(message)s')


def encode_and_save_batch(vae: WanVAE, clip: Optional[CLIPModel], batch: list[ItemInfo]):
    # ... (tu l�gica de encode_and_save_batch SIN CAMBIOS) ...
    # (Solo aseg�rate de que ItemInfo est� correctamente disponible si la type hint es estricta)
    contents = torch.stack([torch.from_numpy(item.content) for item in batch])
    if len(contents.shape) == 4:
        contents = contents.unsqueeze(1)  # B, H, W, C -> B, F, H, W, C

    contents = contents.permute(0, 4, 1, 2, 3).contiguous()  # B, C, F, H, W
    contents = contents.to(vae.device, dtype=vae.dtype)
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    h, w = contents.shape[3], contents.shape[4]
    if h < 8 or w < 8:
        item = batch[0]  # other items should have the same size
        raise ValueError(f"Image or video size too small: {item.item_key} and {len(batch) - 1} more, size: {item.original_size}")

    with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
        latent = vae.encode(contents)
    latent = torch.stack(latent, dim=0)
    latent = latent.to(vae.dtype)

    if clip is not None:
        images = contents[:, :, 0:1, :, :]
        with torch.amp.autocast(device_type=clip.device.type, dtype=torch.float16), torch.no_grad():
            clip_context = clip.visual(images)
        clip_context = clip_context.to(torch.float16)

        B, _, _, lat_h, lat_w = latent.shape
        F = contents.shape[2]
        msk = torch.ones(1, F, lat_h, lat_w, dtype=vae.dtype, device=vae.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)
        msk = msk.repeat(B, 1, 1, 1, 1)

        padding_frames = F - 1
        images_resized = torch.concat([images, torch.zeros(B, 3, padding_frames, h, w, device=vae.device)], dim=2)
        with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
            y = vae.encode(images_resized)
        y = torch.stack(y, dim=0)
        y = y[:, :, :F]
        y = y.to(vae.dtype)
        y = torch.concat([msk, y], dim=1)
    else:
        clip_context = None
        y = None

    if batch[0].control_content is not None:
        control_contents = torch.stack([torch.from_numpy(item.control_content) for item in batch])
        if len(control_contents.shape) == 4:
            control_contents = control_contents.unsqueeze(1)
        control_contents = control_contents.permute(0, 4, 1, 2, 3).contiguous()
        control_contents = control_contents.to(vae.device, dtype=vae.dtype)
        control_contents = control_contents / 127.5 - 1.0
        with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
            control_latent = vae.encode(control_contents)
        control_latent = torch.stack(control_latent, dim=0)
        control_latent = control_latent.to(vae.dtype)
    else:
        control_latent = None

    for i, item in enumerate(batch):
        l = latent[i]
        cctx = clip_context[i] if clip is not None else None
        y_i = y[i] if clip is not None else None
        control_latent_i = control_latent[i] if control_latent is not None else None
        save_latent_cache_wan(item, l, cctx, y_i, control_latent_i)


# --- �NICA FUNCI�N MAIN ---
def main(args_ns: argparse.Namespace): # Renombrar a args_ns para consistencia con el nodo
    node_name_print = "[wan_cache_latents.main]" # Para logs
    logger.info(f"{node_name_print} Executed with args: {vars(args_ns)}")

    device_str = args_ns.device if args_ns.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"{node_name_print} Using device: {device}")

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"{node_name_print} Load dataset config from {args_ns.dataset_config}")
    user_config = cfgutils.load_user_config(args_ns.dataset_config)
    # args_ns debe tener todos los campos que BlueprintGenerator espera de los args del parser
    blueprint = blueprint_generator.generate(user_config, args_ns, architecture=ARCHITECTURE_WAN)
    train_dataset_group = cfgutils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets
    logger.info(f"{node_name_print} Loaded {len(datasets)} datasets.")

    # Esta secci�n de debug_mode parece �til, la mantenemos.
    # args_ns.debug_mode ahora es un string "None" o un string con el modo.
    # cache_latents.show_datasets necesitar�a manejar "None" como no activar el modo debug.
    # O podr�amos hacer:
    # actual_debug_mode = args_ns.debug_mode if args_ns.debug_mode != "None" else None
    # if actual_debug_mode is not None:
    if args_ns.debug_mode != "None": # Asumimos que "None" (string) significa no debug
        logger.info(f"{node_name_print} Debug mode active: {args_ns.debug_mode}. Showing datasets and exiting.")
        # Aseg�rate de que base_cache_latents_script.show_datasets es la funci�n correcta
        base_cache_latents_script.show_datasets(
            datasets, 
            args_ns.debug_mode, # El modo real
            args_ns.console_width, 
            args_ns.console_back if args_ns.console_back != "None" else None, # Pasar None si es "None"
            int(args_ns.console_num_images) if args_ns.console_num_images != "None" else 0, # Pasar int o 0
            fps=16
        )
        return # Terminar si estamos en modo debug de mostrar datasets

    if not hasattr(args_ns, 'vae') or args_ns.vae is None or args_ns.vae == "None": # "None" string por nuestra correcci�n
        logger.error(f"{node_name_print} VAE checkpoint path (args.vae) is required but not provided or is 'None'.")
        raise ValueError("VAE checkpoint is required.")
    
    vae_path = args_ns.vae
    logger.info(f"{node_name_print} Loading VAE model from {vae_path}")
    vae_dtype_str = args_ns.vae_dtype if hasattr(args_ns, 'vae_dtype') and args_ns.vae_dtype is not None else "bfloat16"
    vae_dtype = str_to_dtype(vae_dtype_str) # Convertir string a torch.dtype
    
    vae_cache_cpu_flag = args_ns.vae_cache_cpu if hasattr(args_ns, 'vae_cache_cpu') else False
    cache_device_for_vae = torch.device("cpu") if vae_cache_cpu_flag else None # Ajuste aqu�
    
    vae = WanVAE(vae_path=vae_path, device=device, dtype=vae_dtype, cache_device=cache_device_for_vae)
    logger.info(f"{node_name_print} VAE loaded successfully.")

    clip_model_instance = None # Renombrar para evitar confusi�n con el m�dulo CLIPModel
    # args_ns.clip ahora es un string "None" o una ruta
    if hasattr(args_ns, 'clip') and args_ns.clip is not None and args_ns.clip != "None":
        logger.info(f"{node_name_print} Loading CLIP model from {args_ns.clip}")
        # Aseg�rate de que wan_i2v_14B.i2v_14B est� disponible y tiene "clip_dtype"
        clip_dtype_config = wan_i2v_14B.i2v_14B.get("clip_dtype", torch.float16) # Default si no est�
        clip_model_instance = CLIPModel(dtype=clip_dtype_config, device=device, weight_path=args_ns.clip)
        logger.info(f"{node_name_print} CLIP model loaded successfully.")
    else:
        logger.info(f"{node_name_print} No CLIP model path provided or path is 'None'. Skipping CLIP loading.")


    # Define la funci�n de codificaci�n que se pasar� al script base
    def encode_batch_for_base_script(one_batch: list[ItemInfo]):
        # Llama a tu funci�n local encode_and_save_batch
        encode_and_save_batch(vae, clip_model_instance, one_batch)

    logger.info(f"{node_name_print} Starting dataset encoding process using base script...")
    # Llama a la funci�n encode_datasets del script base
    # Aseg�rate de que base_cache_latents_script est� correctamente importado
    # y que args_ns contiene 'comfy_pbar' si base_cache_latents_script.encode_datasets lo espera.
    base_cache_latents_script.encode_datasets(datasets, encode_batch_for_base_script, args_ns)
    
    logger.info(f"{node_name_print} Latent caching process complete.")


# --- Parser setup (solo para ejecuci�n standalone, no usado por ComfyUI directamente) ---
def wan_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="WAN Latent Caching Specific Arguments")
    parser.add_argument("--vae_cache_cpu", action="store_true", help="cache features in VAE on CPU")
    parser.add_argument("--clip", type=str, default=None, help="CLIP checkpoint path")
    # print(f"[wan_cache_latents.py wan_setup_parser] Arguments added/updated in parser: {id(parser)}") # Opcional
    return parser

# Este bloque if __name__ == "__main__": es solo para ejecutar el script de forma independiente.
# ComfyUI llamar� a la funci�n main(args_ns) directamente.
if __name__ == "__main__":
    # Para la ejecuci�n standalone, necesitamos un parser base com�n.
    # Asumimos que base_cache_latents_script.setup_parser_common() existe.
    if hasattr(base_cache_latents_script, 'setup_parser_common'):
        standalone_parser = base_cache_latents_script.setup_parser_common()
        standalone_parser = wan_setup_parser(standalone_parser) # A�adir args espec�ficos de WAN
        parsed_args = standalone_parser.parse_args()
        main(parsed_args) # Llamar a la funci�n main unificada
    else:
        print("Error: base_cache_latents_script.setup_parser_common not found. Cannot run standalone.")