# File: ComfyUI/custom_nodes/musubi-tuner/wan_trainer_nodes.py
# Version: DEBUG_STEP_22 (Fix arg construction and batch counting) + Bridging TOML + Output Order Fix

import os
import comfy.utils
import folder_paths
import toml
import time
import traceback
import argparse
import subprocess
import shutil
import sys
import re
import importlib
try:
    from .musubi_utils import models_combo, vaes_combo, encoders_combo, clip_vision_files
except Exception:
    from musubi_utils import models_combo, vaes_combo, encoders_combo, clip_vision_files



user_lib_wan_cache_latents = None
user_lib_cache_latents_base = None
user_lib_wan_cache_text_encoder_outputs = None
user_lib_cache_text_encoder_outputs_base = None
user_lib_wan_train_network = None # Still dummy
user_lib_wan_configs = None
user_lib_architecture_wan = "wan"
user_lib_architecture_hv = "hv"
# Import dataset utilities to be used by nodes for batch counting
user_lib_config_utils = None

# Compatibility: handle package/loader differences robustly
try:
    _pkg_dir = os.path.dirname(__file__)
    _parent_dir = os.path.dirname(_pkg_dir)
    for _p in (_pkg_dir, _parent_dir):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)
    _pkg_name = os.path.basename(_pkg_dir)
except Exception:
    _pkg_name = None

try:
    # Try relative import first (when loaded as a ComfyUI custom node)
    try:
        from . import wan_train_network as wtn
    except ImportError:
        # Fall back to absolute import
        import wan_train_network as wtn
    user_lib_wan_train_network = wtn

    try:
        from . import wan_cache_latents as wcl
    except ImportError:
        import wan_cache_latents as wcl
    user_lib_wan_cache_latents = wcl

    try:
        from . import cache_latents as cl_base
    except ImportError:
        import cache_latents as cl_base
    user_lib_cache_latents_base = cl_base

    try:
        from . import wan_cache_text_encoder_outputs as wcteo
    except ImportError:
        import wan_cache_text_encoder_outputs as wcteo
    user_lib_wan_cache_text_encoder_outputs = wcteo

    try:
        from . import cache_text_encoder_outputs as cteo_base
    except ImportError:
        import cache_text_encoder_outputs as cteo_base
    user_lib_cache_text_encoder_outputs_base = cteo_base

    try:
        from .wan import configs as wan_configs_module
    except ImportError:
        from wan import configs as wan_configs_module
    user_lib_wan_configs = wan_configs_module

    try:
        from .dataset import config_utils as dcu_module
    except ImportError:
        from dataset import config_utils as dcu_module
    user_lib_config_utils = dcu_module

    try:
        arch_mod = importlib.import_module(f"{_pkg_name}.dataset.image_video_dataset") if _pkg_name else None
    except Exception:
        arch_mod = None
    if arch_mod is None:
        try:
            from .dataset import image_video_dataset as arch_mod
        except Exception:
            import dataset.image_video_dataset as arch_mod
    ARCHITECTURE_WAN = getattr(arch_mod, 'ARCHITECTURE_WAN', 'wan')
    ARCHITECTURE_HUNYUAN_VIDEO = getattr(arch_mod, 'ARCHITECTURE_HUNYUAN_VIDEO', 'hv')
    user_lib_architecture_wan = ARCHITECTURE_WAN
    user_lib_architecture_hv = ARCHITECTURE_HUNYUAN_VIDEO

except ImportError as e:
    print(f"[MusubiTuner Nodes] FAILED (ImportError) during initial library imports: {e}")
    print(traceback.format_exc())
except Exception as e:
    print(f"[MusubiTuner Nodes] FAILED (Other Exception) during initial library imports: {e}")
    print(traceback.format_exc())
#print("[MusubiTuner Nodes] Initial library import attempts complete.")



# MODELS --------------------------------------------------------------------
MODEL_EXTENSIONS = ['.pth', '.safetensors']
model_folders = folder_paths.get_folder_paths("diffusion_models")
all_diffusion_files = []
# Escanea cada carpeta configurada
for folder in model_folders:
    if os.path.isdir(folder):
        try:
            # Itera sobre los contenidos de la carpeta
            for item_name in os.listdir(folder):
                item_path = os.path.join(folder, item_name)
                # Verifica si es un archivo y tiene una de las extensiones deseadas
                if os.path.isfile(item_path) and any(item_name.lower().endswith(ext) for ext in MODEL_EXTENSIONS):
                    all_diffusion_files.append(item_name)
        except Exception as e:
            print(f"ComfyUI Node ERROR al escanear '{folder}': {e}")

# Elimina duplicados y ordena la lista de nombres de archivo
unique_models = sorted(list(set(all_diffusion_files)))
models_combo = [""] + unique_models
# VAE-------------------------------------------------------------------------
VAE_EXTENSIONS = ['.pth', '.safetensors']
vae_folders = folder_paths.get_folder_paths("vae")
all_vae_files = []
# Escanea cada carpeta configurada
for folder in vae_folders:
    if os.path.isdir(folder):
        try:
            # Itera sobre los contenidos de la carpeta
            for item_name in os.listdir(folder):
                item_path = os.path.join(folder, item_name)
                # Verifica si es un archivo y tiene una de las extensiones deseadas
                if os.path.isfile(item_path) and any(item_name.lower().endswith(ext) for ext in VAE_EXTENSIONS):
                    all_vae_files.append(item_name)
        except Exception as e:
            print(f"ComfyUI Node ERROR al escanear '{folder}': {e}")

# Elimina duplicados y ordena la lista de nombres de archivo
unique_vaes = sorted(list(set(all_vae_files)))
# A�ade el elemento vac�o al principio para el combo
vaes_combo = [""] + unique_vaes
#----------------------------------------------------------------------------------
# TEXT ENCODER-------------------------------------------------------------------------
ENCODER_EXTENSIONS = ['.pth', '.safetensors']
encoder_folders = folder_paths.get_folder_paths("text_encoders")
all_encoder_files = []
# Escanea cada carpeta configurada
for folder in encoder_folders:
    if os.path.isdir(folder):
        try:
            # Itera sobre los contenidos de la carpeta
            for item_name in os.listdir(folder):
                item_path = os.path.join(folder, item_name)
                # Verifica si es un archivo y tiene una de las extensiones deseadas
                if os.path.isfile(item_path) and any(item_name.lower().endswith(ext) for ext in ENCODER_EXTENSIONS):
                    all_encoder_files.append(item_name)
        except Exception as e:
            print(f"ComfyUI Node ERROR al escanear '{folder}': {e}")

# Elimina duplicados y ordena la lista de nombres de archivo
unique_encoders = sorted(list(set(all_encoder_files)))
# A�ade el elemento vac�o al principio para el combo
encoders_combo = [""] + unique_encoders
#----------------------------------------------------------------------------------

class ArgsNamespace(argparse.Namespace):
    def __init__(self, **kwargs): super().__init__(); [setattr(self, k, v) for k, v in kwargs.items()]

class WanDatasetConfig: # This node is working, keep as is
    NODE_NAME = "WanDatasetConfig"; IS_OUTPUT_NODE = False
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"output_filename_prefix": ("STRING", {"default": "musubi_dataset_config", "tooltip": "Generates the training dataset with the specified settings, this will set the training resolution. Override is needed to swap the 512 default res."}),
                             "general_resolution_width": ("INT", {"default": 512, "tooltip": "The general width for training images. This is the base resolution for your dataset, especially when buckets are disabled or for initial scaling."}),
                             "general_resolution_height": ("INT", {"default": 512, "tooltip": "The general height for training images. This is the base resolution for your dataset, especially when buckets are disabled or for initial scaling."}),
                             "general_caption_extension": ("STRING", {"default": ".txt", "tooltip": "The file extension for your caption (description) files (e.g., '.txt'). These files contain the text describing each image in your dataset."}),
                             "general_batch_size": ("INT", {"default": 1, "tooltip": "The number of images processed simultaneously by the model during training. A higher batch size can speed up training but requires more VRAM."}, ),
                             "general_enable_bucket": ("BOOLEAN", {"default": True, "tooltip": "If enabled, images will be grouped into 'buckets' (different resolutions with similar aspect ratios) to optimize VRAM usage and prevent image distortion during training. Recommended to keep enabled."}),
                             "general_bucket_no_upscale": ("BOOLEAN", {"default": False, "tooltip": "If 'Enable Bucket' is active and this is true, images will only be downscaled to fit a bucket's resolution, never upscaled. This can prevent blurry images but might make small images too small."}),
                             "dataset1_image_directory": ("STRING", {"default": "image_text_path", "tooltip": "If 'Enable Bucket' is active and this is true, images will only be downscaled to fit a bucket's resolution, never upscaled. This can prevent blurry images but might make small images too small."}),
                             "dataset1_cache_directory": ("STRING", {"default": "cache_path", "tooltip": "The path to the directory containing the images for this specific dataset. Each image should have a corresponding caption file in the same directory."}),
                             "dataset1_num_repeats": ("INT", {"default": 1, "tooltip": "The path to the directory where cached versions of the processed dataset (e.g., VAE latents, text embeddings) will be stored. Speeds up subsequent training runs."}),
                             "dataset1_override_resolution": ("BOOLEAN", {"default": False, "tooltip": "If enabled, this dataset will use its own specific resolution (`dataset1_resolution_width/height`) instead of the `general_resolution_width/height`."}),
                             "dataset1_resolution_width": ("INT", {"default": 512, "tooltip": "The specific width for images in this dataset if 'Override Resolution' is enabled. Otherwise, the general resolution will be used."}),
                             "dataset1_resolution_height": ("INT", {"default": 512, "tooltip": "The specific height for images in this dataset if 'Override Resolution' is enabled. Otherwise, the general resolution will be used."}),
                            }
               }
    RETURN_TYPES = ("STRING", "*",);
    RETURN_NAMES = ("dataset_toml_path", "trigger_out",);
    FUNCTION = "generate_config";
    CATEGORY = "musubi-tuner/wan/config"
    def generate_config(self, output_filename_prefix, general_resolution_width, general_resolution_height, general_caption_extension, general_batch_size, general_enable_bucket, general_bucket_no_upscale, dataset1_image_directory, dataset1_cache_directory, dataset1_num_repeats, dataset1_override_resolution, dataset1_resolution_width, dataset1_resolution_height):
        node_name_print = f"[MusubiTuner {self.NODE_NAME}]"; print(f"{node_name_print} generate_config called.");
        config_dict = {"general": {"resolution": [general_resolution_height, general_resolution_width],"caption_extension": general_caption_extension, "batch_size": general_batch_size,"enable_bucket": general_enable_bucket, "bucket_no_upscale": general_bucket_no_upscale},"datasets": []};
        dataset1_config_entry = {"image_directory": dataset1_image_directory.replace("\\", "/"),"cache_directory": dataset1_cache_directory.replace("\\", "/"),"num_repeats": dataset1_num_repeats, "enable_bucket": general_enable_bucket, "bucket_no_upscale": general_bucket_no_upscale};_ = dataset1_override_resolution and dataset1_config_entry.update({"resolution": [dataset1_resolution_height, dataset1_resolution_width]}); config_dict["datasets"].append(dataset1_config_entry); output_main_dir = folder_paths.get_output_directory(); node_output_dir = os.path.join(output_main_dir, "musubi_tuner_configs");
        if not os.path.exists(node_output_dir):
            try: os.makedirs(node_output_dir)
            except Exception as e_mkdir: print(f"{node_name_print} Error creating dir: {str(e_mkdir)}"); traceback.print_exc(); return (f"ERROR_DIR", "config_error")
        timestamp_str = time.strftime("%Y%m%d-%H%M%S"); filename = f"{output_filename_prefix}_{timestamp_str}.toml"; filepath = os.path.join(node_output_dir, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f: toml.dump(config_dict, f)
            print(f"{node_name_print} TOML saved: {filepath}"); return (filepath, "config_ok",)
        except Exception as e: print(f"{node_name_print} Error saving TOML: {str(e)}"); traceback.print_exc(); return (f"ERROR_TOML", "config_error")

class WanCacheLatents:
    NODE_NAME = "WanCacheLatents"; IS_OUTPUT_NODE = False
    @classmethod
    def INPUT_TYPES(cls):
        #vae_files = [""] + folder_paths.get_filename_list("vae");
        clip_vision_files = [""] + folder_paths.get_filename_list("clip_vision")
        return {"required": {"dataset_config_toml": ("STRING", {"forceInput": True}),
                             "trigger_in": ("*", {"forceInput": True}),
                             "vae_name": (vaes_combo, {"tooltip": "VAE Model (Variational AutoEncoder). Select the VAE that is compatible with your base model. The VAE is like the model's \"eyes,\" helping to encode and decode images during the process."}),
                             "vae_dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16", "tooltip": "The data precision type for the VAE (the component that handles images). `bfloat16` is a good option for its balance between speed and quality."}),
                             "device": (["cuda", "cpu"], {"default": "cuda", "tooltip": "Choose the hardware device to use for processing. `cuda` uses your NVIDIA GPU (much faster), while `cpu` uses your computer's main processor (much slower)."}),
                             "batch_size_override": ("INT", {"default": 0, "tooltip": "Override the default batch size used for processing data. `0` means use the default value from the dataset configuration. A higher number processes more items at once, but uses more memory."}),
                             "num_workers_override": ("INT", {"default": 0, "tooltip": "Override the default number of workers for loading data. `0` means use the default value from the dataset configuration. A higher number can speed up data loading, but consumes more CPU and RAM."}),
                             "skip_existing": ("BOOLEAN", {"default": False, "tooltip": "If enabled, the node will skip processing files that already have existing cached versions, saving time and computation."}),
                             "keep_cache": ("BOOLEAN", {"default": True, "tooltip": "If enabled, the processed data will be kept in the cache directory even after this operation completes, so it can be reused later without reprocessing."}),
                             "vae_cache_cpu": ("BOOLEAN", {"default": False, "tooltip": "If enabled, the VAE's processed outputs will be cached in your system's main memory (RAM) instead of GPU memory. Useful for saving VRAM, especially with large datasets, but can be slower if you have a slow CPU or hard drive."}),
                             "clip_name": (["None"] + clip_vision_files, {"default": "None", "tooltip": "CLIP Model (Vision). If your training is not for \"Image-to-Video\" (I2V) models, leave it as `None`. If it is I2V, select the appropriate CLIP model, as it helps the model understand the visual information from the input images."}),

                             }
               }

    # --- MODIFIED: Changed output order to dataset_toml_path, trigger_out, status_message ---
    RETURN_TYPES = ("STRING", "*", "STRING",); # Order: STRING (toml), * (trigger), STRING (status)
    RETURN_NAMES = ("dataset_toml_path", "trigger_out", "status_message",); # Order: toml name, trigger name, status name
    # -----------------------------------------------------------------------------------------

    FUNCTION = "execute_cache_latents";
    CATEGORY = "musubi-tuner/wan/preprocess"

    def _get_argparse_defaults_latent(self):
        node_name_print = f"[MusubiTuner {self.NODE_NAME} - Defaulter]"
        defaults_dict = {}
        try:
            if user_lib_cache_latents_base and hasattr(user_lib_cache_latents_base, 'setup_parser_common') and \
               user_lib_wan_cache_latents and hasattr(user_lib_wan_cache_latents, 'wan_setup_parser'):
                parser = argparse.ArgumentParser(add_help=False)
                parser = user_lib_cache_latents_base.setup_parser_common(parser)
                parser = user_lib_wan_cache_latents.wan_setup_parser(parser)
                dummy_cli_args = ['--dataset_config', 'dummy_config.toml']
                # Add other required args if they don't have defaults in the parser
                if not any(action.dest == "vae" and action.required for action in parser._actions if hasattr(action, 'required')): # VAE might be optional with default=None
                    pass # It's okay if it's not strictly required by parser if node provides it
                defaults_ns, _ = parser.parse_known_args(dummy_cli_args)
                defaults_dict = vars(defaults_ns)
            else: print(f"{node_name_print} Could not load parsers for latent defaults.")
        except Exception as e_parser: print(f"{node_name_print} Error getting defaults from latent parser: {e_parser}"); traceback.print_exc()
        return defaults_dict

    def execute_cache_latents(self, dataset_config_toml, trigger_in, vae_name, vae_dtype, device, batch_size_override, num_workers_override, skip_existing, keep_cache, vae_cache_cpu, clip_name):
        node_name_print = f"[MusubiTuner {self.NODE_NAME}]"; print(f"{node_name_print} execute_cache_latents called.")
        # --- MODIFIED: Return dataset_config_toml, trigger_out, status_message on error/skip ---
        if "error" in str(trigger_in).lower() or "ERROR" in str(dataset_config_toml): return (dataset_config_toml, "error_trigger", f"Skipped: prev error")
        if user_lib_wan_cache_latents is None or user_lib_config_utils is None: # Check also config_utils
            return (dataset_config_toml, "error_trigger", "ERROR: Core library modules (wan_cache_latents or config_utils) not loaded.")

        vae_path = folder_paths.get_full_path("vae", vae_name) if vae_name else None; clip_path = folder_paths.get_full_path("clip_vision", clip_name) if clip_name else None
        if not vae_path or not os.path.exists(vae_path): return (dataset_config_toml, "error_trigger", f"ERROR: VAE not found: {vae_name}")
        # -------------------------------------------------------------------------------------

        args_dict = self._get_argparse_defaults_latent()
        args_dict.update({"dataset_config": dataset_config_toml, "vae": vae_path, "vae_dtype": vae_dtype, "device": device, "skip_existing": skip_existing, "keep_cache": keep_cache, "vae_cache_cpu": vae_cache_cpu, "clip": clip_path, "architecture": user_lib_architecture_wan, "debug_dataset": args_dict.get("debug_dataset", False)})

        if batch_size_override > 0: args_dict["batch_size"] = batch_size_override
        if num_workers_override > 0: args_dict["num_workers"] = num_workers_override

        args = ArgsNamespace(**args_dict);
        #print(f"{node_name_print} Constructed args: {vars(args)}")



        #probar la teoria de eliminar esto con el nuevo run......
        # --- NUEVO BUCLE PARA CORREGIR TODOS LOS None ---
        args = ArgsNamespace(**args_dict)
        #print(f"{node_name_print} Constructed args (pre-None fix): {vars(args)}")

        # --- Enfoque Espec�fico (Actualizado) ---
        if hasattr(args, 'debug_mode') and args.debug_mode is None:
            args.debug_mode = "None"
            #print(f"{node_name_print} Converted args.debug_mode to 'None'")
        if hasattr(args, 'console_back') and args.console_back is None:
            args.console_back = "None"
            #print(f"{node_name_print} Converted args.console_back to 'None'")
        if hasattr(args, 'clip') and args.clip is None: # This might be redundant with the path check above, but harmless
            args.clip = "None"
            #print(f"{node_name_print} Converted args.clip to 'None'")

        # --- TRATAMIENTO PARA console_num_images ---
        if hasattr(args, 'console_num_images') and args.console_num_images is None:
            args.console_num_images = 0 # Asignar un entero por defecto
            #print(f"{node_name_print} Converted args.console_num_images to 0")
        # O si supieras que -1 es un valor especial v�lido:
        # args.console_num_images = -1
        #print(f"{node_name_print} Cache latents args post fix: {vars(args)}")


        actual_total_batches = 0
        try:
            print(f"{node_name_print} Attempting to calculate total batches for progress bar...")
            temp_bp_gen = user_lib_config_utils.BlueprintGenerator(user_lib_config_utils.ConfigSanitizer()) # Use imported module
            temp_user_config = user_lib_config_utils.load_user_config(args.dataset_config)
            temp_blueprint = temp_bp_gen.generate(temp_user_config, args, architecture=args.architecture)
            temp_dataset_group = user_lib_config_utils.generate_dataset_group_by_blueprint(temp_blueprint.dataset_group, training=False)
            num_dataset_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
            for dataset_in_group in temp_dataset_group.datasets:
                count = 0;
                for _ in dataset_in_group.retrieve_latent_cache_batches(num_dataset_workers): count += 1
                actual_total_batches += count
            if actual_total_batches == 0 and len(temp_dataset_group.datasets) > 0: actual_total_batches = len(temp_dataset_group.datasets)
            elif actual_total_batches == 0: actual_total_batches = 10
        except Exception as e_count: print(f"{node_name_print} Error calculating batches: {e_count}"); traceback.print_exc(); actual_total_batches = 100

        print(f"{node_name_print} Initializing ProgressBar with total_batches: {actual_total_batches}")
        pbar = comfy.utils.ProgressBar(actual_total_batches); args.comfy_pbar = pbar
        status_message = f"Starting Wan latent caching..."; print(f"{node_name_print} {status_message}")
        try:
            user_lib_wan_cache_latents.main(args)
            status_message = "Wan latent caching completed."; print(f"{node_name_print} {status_message}")
            # --- MODIFIED: Return dataset_config_toml, trigger_out, status_message ---
            return (dataset_config_toml, "latents_ok", status_message,)
            # ------------------------------------------------------------------------
        except SystemExit as e_sysexit:
            error_message = f"Argparse error: SystemExit {e_sysexit.code}"; print(f"{node_name_print} {error_message}")
            # --- MODIFIED: Return dataset_config_toml, trigger_out, status_message ---
            return (dataset_config_toml, "error_trigger", error_message)
            # ------------------------------------------------------------------------
        except Exception as e:
            error_message = f"Error during Wan latent caching: {str(e)}"; print(f"{node_name_print} {error_message}"); print(traceback.format_exc())
            # --- MODIFIED: Return dataset_config_toml, trigger_out, status_message ---
            return (dataset_config_toml, "error_trigger", error_message)
            # ------------------------------------------------------------------------


class WanCacheTextEncoder:  # REAL LOGIC
    NODE_NAME = "WanCacheTextEncoder"; IS_OUTPUT_NODE = False
    @classmethod
    def INPUT_TYPES(cls):
        #text_encoder_files = [""] + folder_paths.get_filename_list("text_encoders")
        return {"required": {"dataset_config_toml": ("STRING", {"forceInput": True}),
                             "trigger_in": ("*", {"forceInput": True}),
                             "t5_name": (encoders_combo, {"tooltip": "Text Encoder (T5). Select the T5 model that your base model uses. This component is the \"translator\" that converts your prompts (text) into a language the image model understands."}),
                             "fp8_t5": ("BOOLEAN", {"default": False, "tooltip": "Enable to use 8-bit precision (FP8) for the T5 text encoder. Saves a lot of memory, but can be experimental and not always stable, and not all GPUs support it."}),
                             "device": (["cuda", "cpu"], {"default": "cuda", "tooltip": "Choose the hardware device to use for processing. `cuda` uses your NVIDIA GPU (much faster), while `cpu` uses your computer's main processor (much slower)."}),
                             "batch_size_override": ("INT", {"default": 0, "tooltip": "Override the default batch size used for processing data. `0` means use the default value from the dataset configuration. A higher number processes more items at once, but uses more memory."}),
                             "num_workers_override": ("INT", {"default": 0, "tooltip": "Override the default number of workers for loading data. `0` means use the default value from the dataset configuration. A higher number can speed up data loading, but consumes more CPU and RAM."}),
                             "skip_existing": ("BOOLEAN", {"default": False, "tooltip": "If enabled, the node will skip processing files that already have existing cached versions, saving time and computation."}),
                             "keep_cache": ("BOOLEAN", {"default": True, "tooltip": "If enabled, the processed data will be kept in the cache directory even after this operation completes, so it can be reused later without reprocessing. If disabled, cached files might be removed."}),
                            }
                }

    # Mantengo el orden modificado del ejemplo anterior: dataset_toml, trigger, status
    RETURN_TYPES = ("STRING", "*", "STRING",); # Order: STRING (toml), * (trigger), STRING (status)
    RETURN_NAMES = ("dataset_toml_path", "trigger_out", "status_message",); # Order: toml name, trigger name, status name

    FUNCTION = "execute_cache_text_encoder";
    CATEGORY = "musubi-tuner/wan/preprocess";


    def _get_argparse_defaults_text(self):
        node_name_print = f"[MusubiTuner {self.NODE_NAME} - Defaulter]"
        defaults_dict = {}
        try:
            if user_lib_cache_text_encoder_outputs_base and hasattr(user_lib_cache_text_encoder_outputs_base, 'setup_parser_common') and \
               user_lib_wan_cache_text_encoder_outputs and hasattr(user_lib_wan_cache_text_encoder_outputs, 'wan_setup_parser'):
                parser = argparse.ArgumentParser(add_help=False)
                parser = user_lib_cache_text_encoder_outputs_base.setup_parser_common(parser)
                parser = user_lib_wan_cache_text_encoder_outputs.wan_setup_parser(parser)
                dummy_cli_args = ['--dataset_config', 'dummy_config.toml', '--t5', 'dummy_t5.pth']
                defaults_ns, _ = parser.parse_known_args(dummy_cli_args)
                defaults_dict = vars(defaults_ns)
            else: print(f"{node_name_print} Could not load parsers for text caching defaults.")
        except Exception as e_parser: print(f"{node_name_print} Error getting defaults from text parser: {e_parser}"); traceback.print_exc()
        return defaults_dict

    def execute_cache_text_encoder(self, dataset_config_toml, trigger_in, t5_name, fp8_t5, device, batch_size_override, num_workers_override, skip_existing, keep_cache):
        node_name_print = f"[MusubiTuner {self.NODE_NAME}]"; print(f"{node_name_print} execute_cache_text_encoder called.")
        # Aseg�rate de que la entrada TOML es v�lida antes de intentar usarla
        if "error" in str(trigger_in).lower() or "skipped" in str(trigger_in).lower() or "ERROR" in str(dataset_config_toml) or not os.path.exists(dataset_config_toml):
             # Devuelve el path TOML recibido, trigger de error, y mensaje de skip
            return (dataset_config_toml, "error_trigger", f"Skipped: prev error/skip or invalid TOML path: {dataset_config_toml}")

        if user_lib_wan_cache_text_encoder_outputs is None or user_lib_config_utils is None:
             # Devuelve el path TOML recibido, trigger de error, y mensaje de error
            return (dataset_config_toml, "error_trigger", "ERROR: Core library modules (wan_cache_text_encoder or config_utils) not loaded.")

        t5_path = folder_paths.get_full_path("text_encoders", t5_name) if t5_name else None
        if not t5_path or not os.path.exists(t5_path):
             # Devuelve el path TOML recibido, trigger de error, y mensaje de error
            return (dataset_config_toml, "error_trigger", f"ERROR: T5 model not found: {t5_name}")

        args_dict = self._get_argparse_defaults_text()
        args_dict.update({"dataset_config": dataset_config_toml, "t5": t5_path, "fp8_t5": fp8_t5, "device": device, "skip_existing": skip_existing, "keep_cache": keep_cache, "architecture": user_lib_architecture_wan, "debug_dataset": args_dict.get("debug_dataset", False)})

        if batch_size_override > 0: args_dict["batch_size"] = batch_size_override
        if num_workers_override > 0: args_dict["num_workers"] = num_workers_override
        args = ArgsNamespace(**args_dict); print(f"{node_name_print} Constructed args: {vars(args)}")

        # --- NUEVO BUCLE PARA CORREGIR TODOS LOS None (Copiado de Latents) ---
        # No es estrictamente necesario re-crear args aqu� si args_dict ya est� completo,
        # pero lo mantengo para consistencia con el c�digo original y el ejemplo de Latents.
        args = ArgsNamespace(**args_dict)
        # print(f"{node_name_print} Constructed args (pre-None fix): {vars(args)}") # Opcional debug

        if hasattr(args, 'debug_mode') and args.debug_mode is None:
            args.debug_mode = "None" # Convertir None a string "None" si el script lo espera as�
            # print(f"{node_name_print} Converted args.debug_mode to 'None'") # Opcional debug
        if hasattr(args, 'console_back') and args.console_back is None:
             args.console_back = "None" # Convertir None a string "None" si el script lo espera as�
             # print(f"{node_name_print} Converted args.console_back to 'None'") # Opcional debug
        # No hay args.clip en este nodo, as� que no hace falta el fix para clip aqu�.

        # TRATAMIENTO PARA console_num_images (Si aplica y existe en el parser base)
        if hasattr(args, 'console_num_images') and args.console_num_images is None:
            args.console_num_images = 0 # Asignar un entero por defecto
            # print(f"{node_name_print} Converted args.console_num_images to 0") # Opcional debug

        # print(f"{node_name_print} Constructed args (post-None fix): {vars(args)}") # Opcional debug
        # -----------------------------------------------------------------------

        # --- INICIO: Calcular el n�mero total de batches para la barra de progreso ---
        actual_total_batches = 0
        try:
            print(f"{node_name_print} Attempting to calculate total batches for progress bar...")
            temp_bp_gen = user_lib_config_utils.BlueprintGenerator(user_lib_config_utils.ConfigSanitizer()) # Use imported module
            temp_user_config = user_lib_config_utils.load_user_config(args.dataset_config)
            temp_blueprint = temp_bp_gen.generate(temp_user_config, args, architecture=args.architecture)
            temp_dataset_group = user_lib_config_utils.generate_dataset_group_by_blueprint(temp_blueprint.dataset_group, training=False)
            num_dataset_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1 if os.cpu_count() else 1)

            for dataset_in_group in temp_dataset_group.datasets:
                count = 0;
                for _ in dataset_in_group.retrieve_text_encoder_output_cache_batches(num_dataset_workers): count += 1
                actual_total_batches += count

            if actual_total_batches == 0 and len(temp_dataset_group.datasets) > 0: actual_total_batches = len(temp_dataset_group.datasets)
            elif actual_total_batches == 0:
                 actual_total_batches = 10
                 print(f"actual batches not calculated correctly, setting up it as 10")
        except Exception as e_count: print(f"{node_name_print} Error calculating batches: {e_count}"); traceback.print_exc(); actual_total_batches = 100


        print(f"{node_name_print} Initializing ProgressBar with total_batches: {actual_total_batches}")
        pbar = comfy.utils.ProgressBar(actual_total_batches); args.comfy_pbar = pbar
        status_message = f"Starting Wan text encoder caching..."; print(f"{node_name_print} {status_message}")
        # --- Fin: Calcular el n�mero total de batches para la barra de progreso ---

        try:
            # La llamada real al script de cach� externo
            user_lib_wan_cache_text_encoder_outputs.main(args)
            #status_message = "Wan text encoder caching completed."; print(f"{node_name_print} {status_message}")
            # Devuelve el path TOML recibido, trigger de �xito, y mensaje de estado
            return (dataset_config_toml, "text_ok", status_message,)
        except SystemExit as e_sysexit:
            error_message = f"Argparse error: SystemExit {e_sysexit.code}"; print(f"{node_name_print} {error_message}")
            # Devuelve el path TOML recibido, trigger de error, y mensaje de error
            return (dataset_config_toml, "error_trigger", error_message)
        except Exception as e:
            error_message = f"Error during Wan text encoder caching: {str(e)}"; print(f"{node_name_print} {error_message}"); print(traceback.format_exc())
            # Devuelve el path TOML recibido, trigger de error, y mensaje de error
            return (dataset_config_toml, "error_trigger", error_message)

class WanLoRATrainer: # DUMMY EXECUTION, REAL INPUTS
    NODE_NAME = "WanLoRATrainer"
    IS_OUTPUT_NODE = True # Correcto, ya que genera un archivo LoRA

    @classmethod
    def INPUT_TYPES(cls):

        wan_tasks_list = list(user_lib_wan_configs.WAN_CONFIGS.keys()) #if user_lib_wan_configs and hasattr(user_lib_wan_configs, 'WAN_CONFIGS') else ["t2v-14B"]
        #diffusion_models_files = [""] + folder_paths.get_filename_list("diffusion_models")
        #vae_files = [""] + folder_paths.get_filename_list("vae")
        #text_encoder_files = [""] + folder_paths.get_filename_list("text_encoders")
        clip_vision_files = [""] + folder_paths.get_filename_list("clip_vision")
        return {
            "required": {
                "dataset_config_toml": ("STRING", {"forceInput": True}), # This input will now come from WanCacheTextEncoder
                "trigger_in": ("*", {"forceInput": True}), # This input will now come from WanCacheTextEncoder
                "dit_name": (models_combo, {"tooltip": "Base Model (DIT). Choose the main model (the large \"skeleton\") on top of which your LoRA will be trained. It's crucial that this is the correct one for your objective."}),
                "vae_name": (vaes_combo, {"tooltip": "VAE Model (Variational AutoEncoder). Select the VAE that is compatible with your base model. The VAE is like the model's \"eyes,\" helping to encode and decode images during the process."}),
                "t5_name": (encoders_combo, {"tooltip": "Text Encoder (T5). Select the T5 model that your base model uses. This component is the \"translator\" that converts your prompts (text) into a language the image model understands."}),
                "clip_name": (["None"] + clip_vision_files, {"default": "None", "tooltip": "CLIP Model (Vision). If your training is not for \"Image-to-Video\" (I2V) models, leave it as `None`. If it is I2V, select the appropriate CLIP model, as it helps the model understand the visual information from the input images."}),
                "output_dir": ("STRING", {"default": os.path.join(folder_paths.get_output_directory(), "musubi_loras") , "tooltip": "The folder where your LoRA will be saved once training is complete. By default, it will be saved in ComfyUI's `output` folder."}),
                "output_name": ("STRING", {"default": "my_wan_lora", "tooltip": "The name you'll give your final LoRA file. For example, if you enter \"my_character\", the file will be named `my_character.safetensors`."}),
                "task": (wan_tasks_list, {"default": "t2v-1.3B", "tooltip": "Defines the type of task you want the model to learn (e.g., generating video from text). This should match the capability of your base model (e.g., t2v-1.3B for Text-to-Video with 1.3 billion parameters)."}),
                "max_train_steps": ("INT", {"default": 2048, "tooltip": "Total number of training \"steps\" or iterations the model will perform. More steps mean longer training, but not always better quality (can lead to \"overfitting\")."}),
                "learning_rate": ("FLOAT", {"default": 2e-4, "step": 0.000001, "tooltip": "The \"speed\" at which the model learns from its mistakes. A very high value can lead to unstable or poor learning; too low, and training will be very slow. This is a very important parameter to adjust."}),
                "optimizer_type": (["AdamW", "AdamW8bit", "Adafactor", "Lion", "Prodigy", "DAdaptAdam"], {"default": "AdamW8bit", "tooltip": "The \"engine\" or algorithm that adjusts the LoRA's weights during training for efficient learning. `AdamW8bit` is a common and memory-efficient option."}),
                "lr_scheduler": (["constant", "constant_with_warmup", "linear", "cosine"], {"default": "cosine", "tooltip": "Defines how the `learning_rate` (learning speed) changes throughout training. For example, `cosine` gradually decreases the speed towards the end, while `constant` keeps it the same."}),
                "network_module": (["networks.lora", "networks.lora_wan"], {"default": "networks.lora_wan", "tooltip": "The specific type of LoRA architecture to be used. For WAN (Wang et al.) models, `networks.lora_wan` is the recommended option."}),
                "network_dim": ("INT", {"default": 32, "tooltip": "Determines the \"power\" or \"complexity\" of your LoRA. A higher value allows for learning more details, but requires more memory (VRAM) and is more prone to overfitting. Also known as `rank`."}),
                "network_alpha": ("FLOAT", {"default": 1.0, "tooltip": "A value that helps balance the impact of `network_dim`. A higher value allows the LoRA to integrate better and not \"deviate\" too much from the base model. Often set to the same value as `network_dim`. Lower than Dim gives more control making it, more strict but less versatility."}),
                "network_dropout": ("FLOAT", {"default": 0.0, "step": 0.1, "tooltip": "A technique to help prevent overfitting (when the model learns the training data too well and doesn't generalize properly). A value between 0.1 and 0.5 can be useful. `0.0` means it's not used."}),
                "mixed_precision": (["no", "fp16", "bf16"], {"default": "bf16", "tooltip": "Allows the use of lower precision numbers (fp16 or bf16) to speed up training and reduce GPU memory (VRAM) usage, at the cost of a minimal quality loss. `bf16` is usually a good balance. `no` uses full precision."}),
                "gradient_accumulation_steps": ("INT", {"default": 1, "tooltip": "Number of training steps the model will wait before updating its weights. This is useful for simulating a larger \"batch size\" (batch of images) if you have low VRAM."}),
                "gradient_checkpointing": ("BOOLEAN", {"default": True, "tooltip": "A technique to save GPU memory (VRAM) during training, at the cost of a slight increase in time. Enable this if you run out of memory."}),
                "seed": ("INT", {"default": 10000, "tooltip": "An initial number that ensures training reproducibility. If you keep it the same and don't change anything else, the training result will be identical every time."}),
                "attn_mode": (["torch", "sdpa", "flash", "flash2", "xformers", "sageattn"], {"default": "sdpa", "tooltip": "The method by which the model handles \"attention\" (how it relates different parts of the data). `sdpa` is a good default value. `Sage` and `xformers` can be faster but require more configuration and hardware/software compatibility (see visual studio note)."}), # Nota: esto se traduce a flags booleanos
                "split_attn": ("BOOLEAN", {"default": False, "tooltip": "Enable to split the attention calculation into smaller parts. Helps save GPU memory, especially useful if `attn_mode` is not enough or if memory is very limited."}),
                "fp8_t5": ("BOOLEAN", {"default": False, "tooltip": "Enable to use 8-bit precision (FP8) for the T5 text encoder. Saves a lot of memory, but can be experimental and not always stable, and not all GPUs support it."}),
                "vae_dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16", "tooltip": "The data precision type for the VAE (the component that handles images). `bfloat16` is a good option for its balance between speed and quality."}),
                "max_data_loader_n_workers": ("INT", {"default": 2, "tooltip": "Number of processes that will load and prepare images for training. A higher value can speed up data loading (especially if you have a lot), but consumes more CPU and RAM."}),
                "persistent_data_loader_workers": ("BOOLEAN", {"default": True, "tooltip": "Whether to keep data loading processes active all the time. Enable this to reduce startup time between \"epochs\" (full passes through the data) if you have many small epochs."}),
                "save_every_n_steps": ("INT", {"default": 0, "tooltip": "Saves a copy (checkpoint) of the LoRA every certain number of training \"steps.\" Useful for saving progress and recovering if something goes wrong. Set to `0` to not save by steps."}),
                "save_every_n_epochs": ("INT", {"default": 1, "tooltip": "Saves a copy (checkpoint) of the LoRA every certain number of \"epochs\" (when it has processed all training data once). Useful for saving stable versions. Set to `0` to not save by epochs."}),
                "save_state": ("BOOLEAN", {"default": False, "tooltip": "Whether to save the full training \"state\" (not just the LoRA file). This allows you to resume training exactly where you left off if it's interrupted or if you want to perform more steps later."}),
                "scale_weight_norms": ("FLOAT", {"default": 1.0, "tooltip": "Adjusts the \"strength\" of the LoRA weights during training. A small value can help keep the LoRA more subtle and closer to the base model. (If set to `0.0`, it means it's not used)."}),
                "timestep_sampling": (["sigma", "uniform", "sigmoid", "shift"], {"default": "shift", "tooltip": "Method for sampling the \"timesteps\" of the diffusion process. This affects how the model learns to generate images at different noise stages. `sigma` is a good default."}),
                "discrete_flow_shift": ("FLOAT", {"default": 3.0, "tooltip": "A specific adjustment for the `shift` timestep sampling method. It's usually not necessary to modify this unless you know what you're doing or are following a very specific guide."}),
                "num_cpu_threads_per_process": ("INT", {"default": 1, "min": 1, "max": os.cpu_count(), "tooltip": "Number of CPU threads per process for accelerate. Defaults to 1. Set to 0 for auto."}),
                "max_train_epochs": ("INT", {"default": 128, "min": 0, "max": 512, "tooltip": "Recomended by Wan, max_train_epochs will override max_train_steps to remove the internal limit of 2048 steps for better learning, max_train_step will be = args.max_train_epochs * math.ceil(len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps) "}),
            },
            "optional": {
                "network_args": ("STRING", {"multiline": True, "default": ""}),
                "optimizer_args": ("STRING", {"multiline": True, "default": ""}),
                "network_weights": ("STRING", {"default": "", "tooltip": "Path to an existing network (e.g., a pre-trained LoRA or a checkpoint) to load its weights. Useful for resuming training or fine-tuning an existing LoRA."}),
                "training_comment": ("STRING", {"multiline": True, "default": ""}),
                "compile_settings": ("DICT", {"forceInput": True, "default": {}}),
                "memory_settings" : ("DICT", {"forceInput": True, "default": {}}),
                "sampling_settings": ("DICT", {"forceInput": True, "default": {}}),
                # --- Recordatorios para futuro nodo Hunyuan y/o logs para wan (comentados) ---
                # "# Hunyuan: lr_warmup_steps_ratio": ("FLOAT", {"default": 0.05}),
                # "# Hunyuan: dit_dtype_override": (["auto", "bfloat16", "float16", "float32"], {"default": "auto"}),
                # "# Hunyuan: text_encoder_dtype": (["bfloat16", "float16", "float32"], {"default": "float16"}),
                # "# Hunyuan: logging_dir_suffix": ("STRING", {"default": "lora_logs"}),
                # "# Hunyuan: log_with": (["tensorboard", "wandb", "all", "none"], {"default": "tensorboard"}),
                # "# Hunyuan: logging_dir": ("STRING", {"default": ""}),
                # "# Hunyuan: min_timestep_train": ("INT", {"default": 0}),
                # "# Hunyuan: max_timestep_train": ("INT", {"default": 1000}),
            }
        }



    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("lora_filepath", "status_message",)
    FUNCTION = "execute_train_lora"
    CATEGORY = "musubi-tuner/wan/train"


    def _get_argparse_defaults_train(self):
        node_name_print = f"[MusubiTuner {self.NODE_NAME} - Defaulter Train]"
        defaults_dict = {}
        try:
            if user_lib_wan_train_network and hasattr(user_lib_wan_train_network, 'get_full_wan_train_arg_parser'):
                parser = user_lib_wan_train_network.get_full_wan_train_arg_parser()
                # No es necesario parsear con dummy_cli_args si solo queremos los defaults
                # Los defaults se obtienen directamente del parser.
                temp_defaults = {}
                for action in parser._actions:
                    if action.dest != "help": # Excluir el argumento de ayuda
                        temp_defaults[action.dest] = action.default

                # Para asegurar que los required que no tienen default en el parser
                # pero s� en el nodo (como dit, vae) se inicializan, aunque no sean usados por el defaulter.
                # El script `get_full_wan_train_arg_parser` deber�a tener defaults o ser opcionales.
                # Si son estrictamente requeridos sin default en el parser, esto podr�a ser un problema,
                # pero el nodo los proveer� despu�s.
                # El objetivo aqu� es obtener los defaults *definidos en el parser*.

                # Forzar algunos valores que el parser podr�a esperar para no fallar si parse_known_args se usa:
                dummy_cli_args_for_parsing_test = [
                     '--output_dir', 'dummy_out_parser_test',
                     '--output_name', 'dummy_lora_parser_test',
                     '--dit', 'dummy_dit.safetensors', # Requerido por el parser base
                     '--vae', 'dummy_vae.safetensors', # Requerido por el parser base
                     '--dataset_config', 'dummy_config.toml' # Requerido por el parser base
                ]
                # Parse known args para obtener una namespace completa con defaults procesados
                # y luego convertirla a dict. Esto es m�s robusto que solo action.default.
                defaults_ns, unknown = parser.parse_known_args(dummy_cli_args_for_parsing_test)
                if unknown: print(f"{node_name_print} Unknown args while getting defaults with dummy args: {unknown}")
                defaults_dict = vars(defaults_ns)

                # Eliminar los valores dummy que no son verdaderos defaults del parser.
                # Los argumentos que s� tienen default en el parser retendr�n ese default.
                # Los que no, y fueron prove�dos por dummy_cli_args, retendr�n ese valor dummy
                # pero ser�n sobrescritos por los valores del nodo.
                # Esto es un poco heur�stico, pero el objetivo es llenar args_dict con los defaults del parser.
                # Los valores de dummy_cli_args_for_parsing_test deben ser reemplazados por los del nodo.

                #print(f"{node_name_print} Defaults from combined train parsers (after dummy parse): {defaults_dict}")
            else:
                print(f"{node_name_print} Could not find get_full_wan_train_arg_parser in wan_train_network.py")
        except Exception as e_parser:
            print(f"{node_name_print} Error getting defaults from train parser: {e_parser}")
            traceback.print_exc()
        return defaults_dict

    def execute_train_lora(self, dataset_config_toml, trigger_in, **kwargs):
        node_name_print = f"[MusubiTuner {self.NODE_NAME}]"
        print(f"{node_name_print} execute_train_lora called. Trigger: {trigger_in}")

        # Gesti�n de salida por defecto si hay error previo
        error_output_dir_default = kwargs.get("output_dir", os.path.join(folder_paths.get_output_directory(), "musubi_loras"))
        error_output_name_default = kwargs.get("output_name", "dummy_error_lora")
        dummy_lora_filepath_on_skip = os.path.join(error_output_dir_default, f"{error_output_name_default}.safetensors")

        if "error" in str(trigger_in).lower() or "skipped" in str(trigger_in).lower() or "ERROR" in str(dataset_config_toml):
            return (dummy_lora_filepath_on_skip, "Skipped: previous error or skip signal")

        if user_lib_wan_train_network is None:
            return (dummy_lora_filepath_on_skip, "ERROR: Core training script (wan_train_network) not loaded.")

        # --- 1. Construir el objeto args ---
        args_dict = self._get_argparse_defaults_train()
        if args_dict is None: args_dict = {} # Salvaguarda


        # --- INICIO DE LA NUEVA L�GICA: Fusionar compile_settings ---
        # Extraer compile_settings de kwargs antes de procesar el resto
        incoming_compile_settings = kwargs.pop("compile_settings", {})
        if incoming_compile_settings:
           print(f"{node_name_print} Received compile_settings: {incoming_compile_settings}")
           args_dict.update(incoming_compile_settings)

        incoming_memory_settings = kwargs.pop("memory_settings", {})

        if incoming_memory_settings:
           print(f"{node_name_print} Received memory_settings: {incoming_memory_settings}") # Corregido el print
           args_dict.update(incoming_memory_settings) # Correcto: Merge memory_settings

        incoming_sampling_settings = kwargs.pop("sampling_settings", {})

        if incoming_sampling_settings:
           print(f"{node_name_print} Received sampling_settings: {incoming_sampling_settings}")
           args_dict.update(incoming_sampling_settings)

        # --- FIN DE LA NUEVA L�GICA ---

        # Sobrescribir y a�adir valores de los inputs del nodo (kwargs)
        # Paths de modelos:
        args_dict["dit"] = folder_paths.get_full_path("diffusion_models", kwargs.get('dit_name'))
        args_dict["vae"] = folder_paths.get_full_path("vae", kwargs.get('vae_name'))
        args_dict["t5"] = folder_paths.get_full_path("text_encoders", kwargs.get('t5_name'))

        clip_name_val = kwargs.get('clip_name')
        args_dict["clip"] = folder_paths.get_full_path("clip_vision", clip_name_val) if clip_name_val and clip_name_val.strip("None") != "" else ""
        args_dict["dataset_config"] = dataset_config_toml # Path al TOML

        # Todos los dem�s kwargs del nodo
        for key, value in kwargs.items():
            if key in ["dit_name", "vae_name", "t5_name", "clip_name", "dataset_config_toml", "trigger_in"]:
                continue # Ya manejados o no son argumentos del script

            if key == "network_args":
                args_dict["network_args"] = [s.strip() for s in value.splitlines() if s.strip()] if value and value.strip() else []
            elif key == "optimizer_args":
                args_dict["optimizer_args"] = [s.strip() for s in value.splitlines() if s.strip()] if value and value.strip() else []
            else:
                args_dict[key] = value

        # Asegurar que output_dir y output_name est�n definidos
        if not args_dict.get("output_dir"):
            args_dict["output_dir"] = error_output_dir_default # Usar el default del nodo
        if not args_dict.get("output_name"):
            args_dict["output_name"] = kwargs.get("output_name", "my_wan_lora_node_default")


        # Manejo de logging_dir (basado en el c�digo original del nodo)
        #log_with_val = args_dict.get("log_with", "none")
        #if log_with_val in ["tensorboard", "all"]:
        #    user_provided_logging_dir = args_dict.get("logging_dir", "") # logging_dir viene de kwargs
        #    if user_provided_logging_dir and os.path.isabs(user_provided_logging_dir):
        #        final_logging_dir = user_provided_logging_dir
        #    else:
        #        base_log_path = args_dict["output_dir"]
        #        log_suffix = args_dict.get("logging_dir_suffix", "logs")
        #        sub_dir_name = user_provided_logging_dir if user_provided_logging_dir else log_suffix
        #        final_logging_dir = os.path.join(base_log_path, sub_dir_name)
        #    args_dict["logging_dir"] = final_logging_dir
        #else:
        #    args_dict["logging_dir"] = None # O el default del parser si es diferente

        # Manejo de attn_mode (convertir a flags booleanos)
        # Esto ya estaba en el c�digo original, se mantiene.
        selected_attn_mode_str = args_dict.get("attn_mode", "torch")
        defaults_from_parser_for_attn = self._get_argparse_defaults_train() # Re-obtener o guardar

        # Asegurar que estos flags existen en args_dict, inicializados a False o al default del parser
        # si el parser los define (lo cual deber�a hacer get_full_wan_train_arg_parser)
        attn_flags = ["sdpa", "flash_attn", "sage_attn", "xformers", "flash3"] # flash3 es hipot�tico
        for flag in attn_flags:
            args_dict[flag] = defaults_from_parser_for_attn.get(flag, False)

        if selected_attn_mode_str == "sdpa": args_dict["sdpa"] = True
        elif selected_attn_mode_str in ["flash", "flash2"]: args_dict["flash_attn"] = True # flash2 usa flash_attn
        elif selected_attn_mode_str == "sageattn": args_dict["sage_attn"] = True
        elif selected_attn_mode_str == "xformers": args_dict["xformers"] = True
        # elif selected_attn_mode_str == "flash3": args_dict["flash3"] = True # Si existiera

        if "attn_mode" in args_dict: # Eliminar la clave original de selecci�n
            del args_dict["attn_mode"]

        # Crear el namespace final a partir de args_dict
        args = ArgsNamespace(**args_dict)

        # Asegurar que el directorio de salida exista
        if not os.path.exists(args.output_dir):
            try:
                os.makedirs(args.output_dir, exist_ok=True)
                print(f"{node_name_print} Created output directory for trainer: {args.output_dir}")
            except Exception as e_mkdir:
                return (os.path.join(args_dict.get("output_dir", error_output_dir_default), f"{args_dict.get('output_name', error_output_name_default)}.safetensors"),
                        f"ERROR_CREATING_TRAIN_DIR: {e_mkdir}")

        # --- IMPRESI�N DETALLADA DE ARGUMENTOS ANTES DE SUBPROCESO ---
        #print(f"{node_name_print} --- ARGUMENTS PREPARED FOR SUBPROCESS ---")
        #for arg_name, arg_value in sorted(vars(args).items()):
        #print(f"{node_name_print}   {arg_name}: {arg_value} (Type: {type(arg_value)})")
        #print(f"{node_name_print} --- END OF PREPARED ARGUMENTS ---")

        # --- INICIO DEL CAMBIO IMPORTANTE ---
        # Si est�s usando shutil.which("accelerate") y ahora te da la ruta correcta:
        accelerate_script_path = shutil.which("accelerate")
        if not accelerate_script_path:
            # Manejar error si no se encuentra, aunque ahora deber�a encontrarlo.
            return (os.path.join(args.output_dir, f"{args.output_name}_accel_error.safetensors"),
                    "ERROR: accelerate executable not found in PATH (ensure ComfyUI env Scripts dir is in PATH).")

        #print(f"{node_name_print} Using accelerate script (from shutil.which): {accelerate_script_path}")
        # --- FIN DEL CAMBIO IMPORTANTE ---

        # Path al script run_wan_training.py (asumiendo que est� en el mismo directorio que este nodo)
        current_node_script_path = os.path.abspath(__file__)
        custom_node_directory = os.path.dirname(current_node_script_path)
        run_script_path = os.path.join(custom_node_directory, "run_wan_training.py")

        if not os.path.exists(run_script_path):
             return (os.path.join(args.output_dir, f"{args.output_name}_script_error.safetensors"),
                    f"ERROR: run_wan_training.py not found at {run_script_path}")

        # Construir el comando para accelerate launch
        cmd = [accelerate_script_path, "launch"] # <--- USA EL SCRIPT DIRECTAMENTE
        # Aqu� se podr�an a�adir argumentos espec�ficos para `accelerate launch` si fuera necesario,
        # por ejemplo, --num_processes, si no se gestiona por config o el script interno.
        # A�adir num_cpu_threads_per_process para accelerate launch
        if hasattr(args, 'num_cpu_threads_per_process') and args.num_cpu_threads_per_process > 0:
            print("SETTING UP CPU THREADS :" + str(args.num_cpu_threads_per_process))
            cmd.extend(["--num_cpu_threads_per_process", str(args.num_cpu_threads_per_process)])
        # El argumento mixed_precision S� es un argumento de `accelerate launch` Y del script.
        # Lo pasaremos al script, y `accelerate launch` tambi�n podr�a recogerlo si es un arg global.
        # Para ser expl�cito con `accelerate launch`:
        if hasattr(args, 'mixed_precision') and args.mixed_precision != "no":
             cmd.extend(["--mixed_precision", args.mixed_precision])
        # Si tuvieras num_gpus o num_processes como input del nodo:
        #cmd.extend(["--num_gpus", str(args.num_gpus)])
        #Max epoch
        if hasattr(args, 'max_train_epochs') and args.max_train_epochs == 0:
             args["max_train_epochs"] = None



    # Aqu� ir�a la l�gica para configurar el modelo o el optimizador para FP8 base
    # Por ejemplo, usar torch.autocast con dtype=torch.float8_e4m3fn si el hardware/PyTorch lo soporta.

        cmd.append(run_script_path)

        # Convertir el namespace 'args' a argumentos de l�nea de comando para run_script_path
        for key, value in vars(args).items():
            if value is None:
                continue # No pasar argumentos con valor None
            if key in ["num_cpu_threads_per_process"]:
                continue
            # Argumentos que son para accelerate launch y ya se han a�adido, o que no son para el script
            if key in ["mixed_precision"]: # mixed_precision se pasa al script Y a accelerate launch
                                           # Si hay otros args solo para accelerate, listarlos aqu� para skip.
                pass                       # Dejar que se a�ada tambi�n para el script, Accelerator lo puede usar.

            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
                # Si es False, se omite el flag (comportamiento est�ndar para store_true)
            elif isinstance(value, list):
                if value: # Solo a�adir si la lista no est� vac�a
                    cmd.append(f"--{key}")
                    for item in value:
                        cmd.append(str(item))
            elif key == "training_comment" and not value.strip(): # No pasar training_comment si est� vac�o
                continue
            elif key == "network_weights" and not value.strip(): # No pasar network_weights_path si est� vac�o
                continue
            else:
                cmd.append(f"--{key}")
                cmd.append(str(value))

        #print(f"{node_name_print} Executing command: {' '.join(cmd)}")


        final_lora_filename = f"{args.output_name}.safetensors" # Asumiendo que el script lo guarda as�
        final_lora_path = os.path.join(args.output_dir, final_lora_filename)

        result = None
        returncode = None # Inicializar returncode

        # --- 3. Ejecutar el entrenamiento en subproceso ---
        try:
            # El CWD para el subproceso podr�a ser importante si hay paths relativos en configs
            # Por defecto es el CWD de ComfyUI. Si el script espera estar en su propio dir:
            # process_cwd = custom_node_directory
            process_cwd = None # Usar CWD actual (ComfyUI base)

            # Capturar stdout y stderr para logging
            # Usar Popen para streaming si es necesario, o run para completado.
            # Para un proceso largo como el entrenamiento, Popen con logs en tiempo real ser�a mejor,
            # pero m�s complejo. `run` es m�s simple para empezar.
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            #print(f"{node_name_print} Setting PYTHONIOENCODING=utf-8 for subprocess.")
            # Cualquier variable de entorno espec�fica aqu�, por ejemplo:
            # env["PYTHONUNBUFFERED"] = "1"
            # env["CUDA_VISIBLE_DEVICES"] = ... (si accelerate no lo gestiona)

            result = subprocess.Popen(
                cmd,
                cwd=process_cwd,
                stdout=subprocess.PIPE,  # Capturar la salida est�ndar
                stderr=subprocess.STDOUT, # Fusionar la salida de error est�ndar en stdout para leer desde un �nico pipe
                text=True,               # Decodificar la salida como texto usando el encoding especificado
                encoding='utf-8',        # Especificar el encoding
                bufsize=1,               # Usar line buffering para leer l�nea a l�nea m�s f�cilmente
                # universal_newlines=True, # Redundante con text=True, pero no hace da�o
                # check=False,           # No es una opci�n de Popen, se maneja al final con wait()
                env=env
            )

            #print(f"{node_name_print} Subprocess started. Output:")

            # Leer la salida en tiempo real, l�nea por l�nea
            # Este bucle se ejecutar� mientras el subproceso produzca salida
            # process.stdout es un objeto tipo archivo que se puede iterar

            actual_progress = 0
            total_progress = 100
            current_line = "";
            digits_string = "";
            check_start_line = False;
            pbar = comfy.utils.ProgressBar(total_progress);
            for line in result.stdout:
                # Imprimir la l�nea inmediatamente. rstrip() elimina el salto de l�nea
                # que ya viene en 'line' del proceso hijo.
                current_line = line.rstrip();
                check_start_line = current_line.startswith('steps');
                if check_start_line:
                   digit_string = re.search(r'(\d+)\%?', current_line)
                   if digit_string:
                      try:
                          actual_progress = int(digit_string.group(1))
                      except ValueError:
                              print(f"Error al convertir '{digit_string}' a entero.")
                   pbar.update_absolute(actual_progress, total_progress)

                print(f"{node_name_print} TRAINER: {current_line}")
                # Forzar que la salida se escriba en la consola inmediatamente
                sys.stdout.flush()




            # Una vez que el bucle termina (porque el pipe se cerr�, indicando que el proceso acab�),
            # esperamos a que el proceso realmente termine y obtenemos su c�digo de salida.
            returncode = result.wait()

            #print(f"{node_name_print} Subprocess finished with return code {returncode}.")

            # Ahora manejamos el c�digo de salida
            if returncode != 0:
                # El subproceso termin� con un error (c�digo != 0)
               error_message = f"Training subprocess failed with return code {returncode}. See logs above for details."
               print(f"{node_name_print} ERROR: {error_message}", file=sys.stderr)
               # Retornar la ruta esperada, pero con un estado de error
               return (final_lora_path, f"ERROR: {error_message}")


            if not os.path.exists(final_lora_path):
               # El archivo no se encontr� a pesar del return code 0. Esto es una advertencia.
               # time.sleep(1) # El sleep del c�digo original estaba aqu�, puedes dejarlo si quieres ser extra precavido con FS
               if not os.path.exists(final_lora_path): # Doble check despu�s del sleep opcional
                   warning_message = f"WARNING: Subprocess finished with return code 0, but final LoRA file not found at expected path: {final_lora_path}. Check subprocess logs for potential warnings/errors during save."
                   print(f"{node_name_print} {warning_message}", file=sys.stderr)
                   # Retornar la ruta esperada, pero con un estado de advertencia
                   return (final_lora_path, f"WARNING: {warning_message}")

            status_message = f"Wan LoRA training via subprocess completed. LoRA saved at: {final_lora_path}"
            print(f"{node_name_print} {status_message}")
            # Retornar la ruta y el estado de �xito
            return (final_lora_path, "train_ok")
        except FileNotFoundError as e_fnf:
            error_message = f"ERROR: Command not found: {cmd[0]}. Ensure accelerate or the training script is installed and in your PATH, or provide full paths. Details: {e_fnf}"
            print(f"{node_name_print} {error_message}", file=sys.stderr)
            # Incluir traceback para depuraci�n
            traceback.print_exc()
            # Retornar una ruta indicativa de error (o la esperada si args est� disponible) y el mensaje de error
            return (os.path.join(args.output_dir, f"{args.output_name}_error_cmd_not_found.safetensors"), error_message)
        except subprocess.TimeoutExpired as e_timeout:
            # Intentar terminar el subproceso si todav�a est� corriendo
            if result and result.poll() is None:
                 print(f"{node_name_print} Subprocess timed out. Attempting to terminate...", file=sys.stderr)
                 try:
                     result.terminate() # O result.kill() para m�s agresividad
                     # Esperar un poco para que el proceso termine despu�s de la se�al
                     result.wait(timeout=5) # Puedes ajustar o quitar el timeout aqu� si prefieres no esperar
                 except:
                     # Ignorar errores al intentar terminar el proceso
                     pass
                 print(f"{node_name_print} Subprocess termination attempt finished.", file=sys.stderr)

            error_message = f"ERROR: Training subprocess timed out after {e_timeout.timeout} seconds. Details: {e_timeout}"
            print(f"{node_name_print} {error_message}", file=sys.stderr)
            traceback.print_exc()
            # Retornar una ruta indicativa de error y el mensaje de error
            return (os.path.join(args.output_dir, f"{args.output_name}_timeout.safetensors"), error_message)
        except Exception as e_general:
        # Intentar terminar el subproceso si parece que se inici� y luego fall�
            if result and result.poll() is None:
                 print(f"{node_name_print} Unexpected exception occurred, subprocess might still be running. Attempting to terminate...", file=sys.stderr)
                 try:
                     result.terminate()
                     result.wait(timeout=5)
                 except:
                     pass
                 print(f"{node_name_print} Subprocess termination attempt finished.", file=sys.stderr)

            error_message = f"ERROR: An unexpected error occurred during subprocess execution: {str(e_general)}"
            print(f"{node_name_print} {error_message}", file=sys.stderr)
            # Incluir traceback para depuraci�n detallada
            traceback.print_exc()
            # Retornar la ruta esperada (aunque probablemente no exista) y el mensaje de error
            return (final_lora_path, f"ERROR: {error_message}")



NODE_CLASS_MAPPINGS = {"WanDatasetConfig": WanDatasetConfig, "WanCacheLatents": WanCacheLatents, "WanCacheTextEncoder": WanCacheTextEncoder, "WanLoRATrainer": WanLoRATrainer}
NODE_DISPLAY_NAME_MAPPINGS = {"WanDatasetConfig": "Musubi Dataset Config (Wan)", "WanCacheLatents": "Musubi Cache Latents (Wan)", "WanCacheTextEncoder": "Musubi Cache Text Embeds (Wan)", "WanLoRATrainer": "Musubi LoRA Trainer (Wan)"}
