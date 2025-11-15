# File: musubi-tuner/wan_cache_text_encoder_outputs.py
# Refactored for ComfyUI node integration
import argparse
import os
from typing import Optional, Union # Keep for type hints, even if not directly used by this script's logic
# Standard library imports
import accelerate # Keep as is
import logging
# import numpy as np # Not directly used in this script's provided logic
import torch
# from tqdm import tqdm # Will be replaced by comfy_pbar or conditional tqdm

# Try relative imports first, fall back to absolute imports
try:
    from .dataset import config_utils
    from .dataset.config_utils import BlueprintGenerator, ConfigSanitizer
    from .dataset.image_video_dataset import ARCHITECTURE_WAN, ItemInfo, save_text_encoder_output_cache_wan
    from .wan.configs import wan_t2v_14B
    from .wan.modules.t5 import T5EncoderModel
    from . import cache_text_encoder_outputs as base_cache_text_encoder_script
except ImportError:
    from dataset import config_utils
    from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
    from dataset.image_video_dataset import ARCHITECTURE_WAN, ItemInfo, save_text_encoder_output_cache_wan
    from wan.configs import wan_t2v_14B
    from wan.modules.t5 import T5EncoderModel
    import cache_text_encoder_outputs as base_cache_text_encoder_script



logger = logging.getLogger(__name__)
# Remove basicConfig if it's handled at a higher level (e.g., in __init__.py of musubi-tuner or by ComfyUI)
# If not, keep it but be mindful of multiple configurations.
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s %(name)s] %(message)s')


def encode_and_save_batch(
    text_encoder: T5EncoderModel, 
    batch: list, # Changed from list[ItemInfo] to list to avoid NameError if ItemInfo isn't fully resolved during initial parse
    device: torch.device, 
    accelerator: Optional[accelerate.Accelerator]
):
    # Type hint for batch items if ItemInfo is confirmed available
    # batch_items: List[ItemInfo] = batch 
    prompts = [item.caption for item in batch]

    with torch.no_grad():
        if accelerator is not None:
            with accelerator.autocast(): # autocast should be on accelerator, not torch.amp directly here
                context = text_encoder(prompts, device)
        else:
            context = text_encoder(prompts, device)

    for item, ctx in zip(batch, context):
        # Assuming ItemInfo type for item if it was correctly imported and available
        save_text_encoder_output_cache_wan(item, ctx) # ItemInfo.text_encoder_output_cache_path will be used here


# --- Main function callable by ComfyUI node ---
def main(args_ns: argparse.Namespace): # Changed 'args' to 'args_ns' to avoid conflict with argparse module
    node_name_print = "[wan_cache_text_encoder_outputs.main]"
    logger.info(f"{node_name_print} Called with args: {vars(args_ns)}")

    device = args_ns.device if args_ns.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"{node_name_print} Using device: {device}")

    # Load dataset config
    logger.info(f"{node_name_print} Loading dataset config from: {args_ns.dataset_config}")
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    user_config = config_utils.load_user_config(args_ns.dataset_config)
    
    # The 'args_ns' passed to generate() is used for fallbacks (e.g., args_ns.debug_dataset)
    # It needs to contain all keys that BlueprintGenerator might access from the 'argparse_config' fallback.
    # architecture=ARCHITECTURE_WAN is correct.
    blueprint = blueprint_generator.generate(user_config, args_ns, architecture=ARCHITECTURE_WAN)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets
    logger.info(f"{node_name_print} Loaded {len(datasets)} datasets.")

    # Define accelerator for fp8 inference
    # Assuming wan_t2v_14B.t2v_14B is correctly imported and has t5_dtype
    t5_config = wan_t2v_14B.t2v_14B 
    accelerator_instance = None # Renamed to avoid conflict with 'accelerate' module
    if args_ns.fp8_t5:
        logger.info(f"{node_name_print} Using FP8 for T5 with mixed_precision.")
        # Ensure t5_config.t5_dtype is a torch.dtype
        mp = "bf16" if t5_config.t5_dtype == torch.bfloat16 else "fp16"
        accelerator_instance = accelerate.Accelerator(mixed_precision=mp)
    
    logger.info(f"{node_name_print} Preparing cache files and paths...")
    all_cache_files_for_dataset, all_cache_paths_for_dataset = \
        base_cache_text_encoder_script.prepare_cache_files_and_paths(datasets)

    # Load T5
    logger.info(f"{node_name_print} Loading T5 from: {args_ns.t5}")
    text_encoder = T5EncoderModel(
        text_len=t5_config.text_len, 
        dtype=t5_config.t5_dtype, 
        device=device, 
        weight_path=args_ns.t5, 
        fp8=args_ns.fp8_t5
    )

    # Encode with T5
    logger.info(f"{node_name_print} Starting encoding with T5...")

    def encode_for_text_encoder_wrapper(batch_items: list): # Renamed to avoid conflict
        # Type hint if ItemInfo is available: batch_items: List[ItemInfo]
        encode_and_save_batch(text_encoder, batch_items, device, accelerator_instance)

    # Pass comfy_pbar from args_ns to process_text_encoder_batches if it's modified to accept it
    # For now, assuming base_cache_text_encoder_script.process_text_encoder_batches handles its own tqdm
    # or needs to be refactored.
    # If you refactor process_text_encoder_batches to take comfy_pbar:
    # comfy_pbar = args_ns.comfy_pbar if hasattr(args_ns, 'comfy_pbar') else None

    base_cache_text_encoder_script.process_text_encoder_batches(
        args_ns,                             # 1. El objeto Namespace completo
        datasets,                            # 2. datasets
        all_cache_files_for_dataset,         # 3. all_cache_files_for_dataset
        all_cache_paths_for_dataset,         # 4. all_cache_paths_for_dataset
        encode_for_text_encoder_wrapper      # 5. la funci�n de codificaci�n espec�fica para WAN
    )
    del text_encoder
    logger.info(f"{node_name_print} Finished encoding with T5.")

    logger.info(f"{node_name_print} Post-processing cache files...")
    base_cache_text_encoder_script.post_process_cache_files(
        datasets, 
        all_cache_files_for_dataset, 
        all_cache_paths_for_dataset,
        args_ns # Pass args_ns for args.keep_cache
    )
    logger.info(f"{node_name_print} Text encoder caching process complete.")


def wan_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Adds WAN-specific arguments for text encoder caching to an existing ArgumentParser object.
    This function will be called by the ComfyUI node to get default argument values.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="WAN Text Encoder Caching Specific Arguments")

    parser.add_argument(
        "--t5", 
        type=str, 
        default=None, # Important for default fetching
        # required=True, # 'required' makes default fetching harder if not provided a dummy
        help="text encoder (T5) checkpoint path"
    )
    parser.add_argument(
        "--fp8_t5", 
        action="store_true", # Default is False
        help="use fp8 for Text Encoder model"
    )
    return parser

# Remove the if __name__ == "__main__": block
# The ComfyUI node will call the main() function directly.
