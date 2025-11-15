# File: musubi-tuner/cache_text_encoder_outputs.py (Base script, refactored)

import argparse
import os
from typing import Optional, Union, List, Set # Explicit List and Set for type hints

# import numpy as np # Not directly used in this script's logic
import torch
# from tqdm import tqdm # Will be replaced or made conditional

import accelerate
import logging

# Try relative imports first, fall back to absolute imports
try:
    from .dataset import config_utils # For load_user_config
    from .dataset.config_utils import BlueprintGenerator, ConfigSanitizer
    from .dataset.image_video_dataset import ARCHITECTURE_HUNYUAN_VIDEO, BaseDataset, ItemInfo, save_text_encoder_output_cache
    # For Hunyuan specific text encoder parts
    try:
        from .hunyuan_model import text_encoder as text_encoder_module
        from .hunyuan_model.text_encoder import TextEncoder
    except ImportError:
        print("[cache_text_encoder_outputs.py] Warning: Could not import Hunyuan text encoder modules. This is fine if only used for WAN.")
        TextEncoder = None # Placeholder if not found
        text_encoder_module = None
except ImportError:
    from dataset import config_utils
    from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
    from dataset.image_video_dataset import ARCHITECTURE_HUNYUAN_VIDEO, BaseDataset, ItemInfo, save_text_encoder_output_cache
    # For Hunyuan specific text encoder parts
    try:
        from hunyuan_model import text_encoder as text_encoder_module
        from hunyuan_model.text_encoder import TextEncoder
    except ImportError:
        print("[cache_text_encoder_outputs.py] Warning: Could not import Hunyuan text encoder modules. This is fine if only used for WAN.")
        TextEncoder = None # Placeholder if not found
        text_encoder_module = None

# Assuming model_utils is in your musubi-tuner/utils directory
# The original import was "from dataset.utils.model_utils import str_to_dtype"
# If model_utils is in musubi-tuner/utils/, the import should be:
try:
    from .train_utils.model_utils import str_to_dtype
except ImportError:
    print("[cache_text_encoder_outputs.py] Warning: Could not import str_to_dtype from .utils.model_utils")
    def str_to_dtype(x): return torch.float32 # Fallback

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s %(name)s] %(message)s')

# --- Functions from your script ---

def encode_prompt(text_encoder: TextEncoder, prompt: Union[str, list[str]]):
    if TextEncoder is None: # Check if TextEncoder was imported
        raise RuntimeError("TextEncoder module (Hunyuan) not available but encode_prompt was called.")
    data_type = "video" 
    text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)
    with torch.no_grad():
        prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type)
    return prompt_outputs.hidden_state, prompt_outputs.attention_mask

def encode_and_save_batch(
    text_encoder: TextEncoder, 
    batch: list, # list[ItemInfo]
    is_llm: bool, 
    accelerator_instance: Optional[accelerate.Accelerator] # Renamed from accelerator
):
    if TextEncoder is None:
         raise RuntimeError("TextEncoder module (Hunyuan) not available but encode_and_save_batch was called.")
    prompts = [item.caption for item in batch]
    if accelerator_instance is not None:
        with accelerator_instance.autocast():
            prompt_embeds, prompt_mask = encode_prompt(text_encoder, prompts)
    else:
        prompt_embeds, prompt_mask = encode_prompt(text_encoder, prompts)
    for item, embed, mask in zip(batch, prompt_embeds, prompt_mask):
        save_text_encoder_output_cache(item, embed, mask, is_llm) # Assumes ItemInfo and this function are correctly imported/defined

def prepare_cache_files_and_paths(datasets: List[BaseDataset]): # Type hint with BaseDataset
    all_cache_files_for_dataset = []
    all_cache_paths_for_dataset = []
    for dataset in datasets:
        # Ensure dataset objects have 'get_all_text_encoder_output_cache_files' method
        if not hasattr(dataset, 'get_all_text_encoder_output_cache_files'):
            logger.warning(f"Dataset {type(dataset).__name__} missing 'get_all_text_encoder_output_cache_files' method.")
            all_cache_files_for_dataset.append(set())
            all_cache_paths_for_dataset.append(set())
            continue
            
        all_cache_files = [os.path.normpath(file) for file in dataset.get_all_text_encoder_output_cache_files()]
        all_cache_files = set(all_cache_files)
        all_cache_files_for_dataset.append(all_cache_files)
        all_cache_paths_for_dataset.append(set())
    return all_cache_files_for_dataset, all_cache_paths_for_dataset

def process_text_encoder_batches(
    args_ns: argparse.Namespace, # Changed to args_ns
    datasets: List[BaseDataset], # Type hint
    all_cache_files_for_dataset: List[Set[str]], # Type hint
    all_cache_paths_for_dataset: List[Set[str]], # Type hint
    encode_function: callable, # Renamed from 'encode'
    # comfy_pbar: Optional[object] = None # Optional comfy_pbar
):
    node_name_print = "[cache_text_encoder_outputs.process_text_encoder_batches]"
    num_workers = args_ns.num_workers if args_ns.num_workers is not None else max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
    
    use_comfy_pbar = hasattr(args_ns, 'comfy_pbar') and args_ns.comfy_pbar is not None
    if use_comfy_pbar:
        logger.info(f"{node_name_print} Using ComfyUI ProgressBar.")
    else:
        from tqdm import tqdm # Fallback to tqdm if no comfy_pbar
        logger.info(f"{node_name_print} ComfyUI ProgressBar not found. Using tqdm if available.")

    total_batches_iterated = 0

    for i, dataset in enumerate(datasets):
        logger.info(f"Encoding dataset [{i}] ({type(dataset).__name__})")
        all_cache_files = all_cache_files_for_dataset[i]
        all_cache_paths = all_cache_paths_for_dataset[i]
        
        # The iterator for batches
        batch_iterator = dataset.retrieve_text_encoder_output_cache_batches(num_workers)
        
        # Wrap with tqdm if comfy_pbar is not used and tqdm is available
        if not use_comfy_pbar and 'tqdm' in globals():
            # To use tqdm, we need to know the number of batches, which might not be easy with a generator.
            # If dataset has a way to get batch count, use it. Otherwise, tqdm might not show total.
            try:
                num_batches_for_dataset = dataset.get_num_text_encoder_batches(num_workers) # You'd need to implement this
                batch_iterator = tqdm(batch_iterator, total=num_batches_for_dataset, desc=f"Dataset {i} batches")
            except AttributeError: # If dataset doesn't have get_num_text_encoder_batches
                batch_iterator = tqdm(batch_iterator, desc=f"Dataset {i} batches (total unknown)")

        for batch_items in batch_iterator: # batch_items is the list[ItemInfo]
            if not batch_items:
                if use_comfy_pbar: args_ns.comfy_pbar.update(1) # Still counts as a processed (empty) batch
                total_batches_iterated += 1
                continue

            all_cache_paths.update([os.path.normpath(item.text_encoder_output_cache_path) for item in batch_items])

            current_batch_to_process = batch_items
            if args_ns.skip_existing:
                filtered_items = [
                    item for item in current_batch_to_process 
                    if not os.path.normpath(item.text_encoder_output_cache_path) in all_cache_files
                ]
                if not filtered_items:
                    if use_comfy_pbar: args_ns.comfy_pbar.update(1) # Update for the skipped batch
                    total_batches_iterated += 1
                    continue
                current_batch_to_process = filtered_items
            
            # Sub-batching for the encode_function
            sub_batch_size = args_ns.batch_size if args_ns.batch_size is not None else len(current_batch_to_process)
            for k in range(0, len(current_batch_to_process), sub_batch_size):
                sub_batch = current_batch_to_process[k : k + sub_batch_size]
                if not sub_batch:
                    continue
                logger.debug(f"{node_name_print} Encoding sub-batch of size {len(sub_batch)} for dataset {i}")
                encode_function(sub_batch)
            
            if use_comfy_pbar:
                args_ns.comfy_pbar.update(1) # Update per processed batch from dataset iterator
            total_batches_iterated +=1

    logger.info(f"{node_name_print} Total batches iterated (for pbar updates): {total_batches_iterated}")


def post_process_cache_files(
    datasets: List[BaseDataset], # Type hint
    all_cache_files_for_dataset: List[Set[str]], # Type hint
    all_cache_paths_for_dataset: List[Set[str]], # Type hint
    args_ns: argparse.Namespace # Pass args_ns for args.keep_cache
):
    for i, dataset in enumerate(datasets):
        all_cache_files = all_cache_files_for_dataset[i]
        all_cache_paths = all_cache_paths_for_dataset[i]
        for cache_file in all_cache_files:
            if cache_file not in all_cache_paths:
                if args_ns.keep_cache:
                    logger.info(f"Keep cache file not in the dataset: {cache_file}")
                else:
                    try:
                        os.remove(cache_file)
                        logger.info(f"Removed old cache file: {cache_file}")
                    except OSError as e_remove:
                         logger.error(f"Error removing old cache file {cache_file}: {e_remove}")

# --- Main function callable by ComfyUI node (via wan_cache_text_encoder_outputs.py) ---
# This main function is for Hunyuan, wan_cache_text_encoder_outputs.py will have its own main.
def main_hunyuan(args_ns: argparse.Namespace): # Renamed to avoid clash if this script is imported
    node_name_print = "[cache_text_encoder_outputs.main_hunyuan]"
    logger.info(f"{node_name_print} Called with args: {vars(args_ns)}")

    device = args_ns.device if args_ns.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"{node_name_print} Using device: {device}")

    if TextEncoder is None or text_encoder_module is None:
        logger.error(f"{node_name_print} Hunyuan TextEncoder modules not available. Cannot proceed for Hunyuan.")
        return

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"{node_name_print} Load dataset config from {args_ns.dataset_config}")
    user_config = config_utils.load_user_config(args_ns.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args_ns, architecture=ARCHITECTURE_HUNYUAN_VIDEO)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    accelerator_instance = None
    if args_ns.fp8_llm: # This arg is from hv_setup_parser
        accelerator_instance = accelerate.Accelerator(mixed_precision="fp16") # fp16 for Hunyuan LLM

    all_cache_files_for_dataset, all_cache_paths_for_dataset = prepare_cache_files_and_paths(datasets)

    text_encoder_dtype = torch.float16 if args_ns.text_encoder_dtype is None else str_to_dtype(args_ns.text_encoder_dtype)
    
    logger.info(f"{node_name_print} loading text encoder 1: {args_ns.text_encoder1}")
    text_encoder_1 = text_encoder_module.load_text_encoder_1(args_ns.text_encoder1, device, args_ns.fp8_llm, text_encoder_dtype)
    text_encoder_1.to(device=device)
    
    logger.info(f"{node_name_print} Encoding with Text Encoder 1")
    def encode_for_te1(batch: list): encode_and_save_batch(text_encoder_1, batch, is_llm=True, accelerator_instance=accelerator_instance)
    process_text_encoder_batches(args_ns, datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, encode_for_te1)
    del text_encoder_1

    logger.info(f"{node_name_print} loading text encoder 2: {args_ns.text_encoder2}")
    text_encoder_2 = text_encoder_module.load_text_encoder_2(args_ns.text_encoder2, device, text_encoder_dtype)
    text_encoder_2.to(device=device)

    logger.info(f"{node_name_print} Encoding with Text Encoder 2")
    def encode_for_te2(batch: list): encode_and_save_batch(text_encoder_2, batch, is_llm=False, accelerator_instance=None)
    process_text_encoder_batches(args_ns, datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, encode_for_te2)
    del text_encoder_2

    post_process_cache_files(datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args_ns)
    logger.info(f"{node_name_print} Hunyuan text encoder caching process complete.")


# --- Parser setup functions ---
def setup_parser_common(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(add_help=False) # add_help=False if chaining
    
    parser.add_argument("--dataset_config", type=str, required=True, help="path to dataset config .toml file")
    parser.add_argument("--device", type=str, default=None, help="device to use")
    parser.add_argument("--batch_size", type=int, default=None, help="batch size for encoding")
    parser.add_argument("--num_workers", type=int, default=None, help="number of workers for dataset processing")
    parser.add_argument("--skip_existing", action="store_true", help="skip existing cache files")
    parser.add_argument("--keep_cache", action="store_true", help="keep cache files not in dataset")
    return parser

def hv_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser: # For Hunyuan
    if parser is None:
         parser = argparse.ArgumentParser(description="Hunyuan Text Encoder Caching Specific Arguments")
    parser.add_argument("--text_encoder1", type=str, required=True, help="Text Encoder 1 directory")
    parser.add_argument("--text_encoder2", type=str, required=True, help="Text Encoder 2 directory")
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="data type for Text Encoder, default is float16")
    parser.add_argument("--fp8_llm", action="store_true", help="use fp8 for Text Encoder 1 (LLM)")
    return parser

# Remove the if __name__ == "__main__": block
