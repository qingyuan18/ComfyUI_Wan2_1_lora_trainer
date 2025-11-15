import argparse
import os
import glob
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import logging

# Try relative imports first, fall back to absolute imports
try:
    from .dataset.config_utils import BlueprintGenerator, ConfigSanitizer # type: ignore
    from .dataset.image_video_dataset import BaseDataset, ItemInfo, save_latent_cache, ARCHITECTURE_HUNYUAN_VIDEO # type: ignore
    from .hunyuan_model.vae import load_vae
    from .hunyuan_model.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
    from .train_utils.model_utils import str_to_dtype
except ImportError:
    from dataset.config_utils import BlueprintGenerator, ConfigSanitizer # type: ignore
    from dataset.image_video_dataset import BaseDataset, ItemInfo, save_latent_cache, ARCHITECTURE_HUNYUAN_VIDEO # type: ignore
    from hunyuan_model.vae import load_vae
    from hunyuan_model.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
    from train_utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def show_image(image: Union[list[Union[Image.Image, np.ndarray], Union[Image.Image, np.ndarray]]]) -> int:
    import cv2

    imgs = (
        [image]
        if (isinstance(image, np.ndarray) and len(image.shape) == 3) or isinstance(image, Image.Image)
        else [image[0], image[-1]]
    )
    if len(imgs) > 1:
        print(f"Number of images: {len(image)}")
    for i, img in enumerate(imgs):
        if len(imgs) > 1:
            print(f"{'First' if i == 0 else 'Last'} image: {img.shape}")
        else:
            print(f"Image: {img.shape}")
        cv2_img = np.array(img) if isinstance(img, Image.Image) else img
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", cv2_img)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if k == ord("q") or k == ord("d"):
            return k
    return k


def show_console(
    image: Union[list[Union[Image.Image, np.ndarray], Union[Image.Image, np.ndarray]]],
    width: int,
    back: str,
    interactive: bool = False,
) -> int:
    from ascii_magic import from_pillow_image, Back

    back = None
    if back is not None:
        back = getattr(Back, back.upper())

    k = None
    imgs = (
        [image]
        if (isinstance(image, np.ndarray) and len(image.shape) == 3) or isinstance(image, Image.Image)
        else [image[0], image[-1]]
    )
    if len(imgs) > 1:
        print(f"Number of images: {len(image)}")
    for i, img in enumerate(imgs):
        if len(imgs) > 1:
            print(f"{'First' if i == 0 else 'Last'} image: {img.shape}")
        else:
            print(f"Image: {img.shape}")
        pil_img = img if isinstance(img, Image.Image) else Image.fromarray(img)
        ascii_img = from_pillow_image(pil_img)
        ascii_img.to_terminal(columns=width, back=back)

        if interactive:
            k = input("Press q to quit, d to next dataset, other key to next: ")
            if k == "q" or k == "d":
                return ord(k)

    if not interactive:
        return ord(" ")
    return ord(k) if k else ord(" ")


def save_video(image: Union[list[Union[Image.Image, np.ndarray], Union[Image.Image, np.ndarray]]], cache_path: str, fps: int = 24):
    import av

    directory = os.path.dirname(cache_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if (isinstance(image, np.ndarray) and len(image.shape) == 3) or isinstance(image, Image.Image):
        # save image
        image_path = cache_path.replace(".safetensors", ".jpg")
        img = image if isinstance(image, Image.Image) else Image.fromarray(image)
        img.save(image_path)
        print(f"Saved image: {image_path}")
    else:
        imgs = image
        print(f"Number of images: {len(imgs)}")
        # save video
        video_path = cache_path.replace(".safetensors", ".mp4")
        height, width = imgs[0].shape[0:2]

        # create output container
        container = av.open(video_path, mode="w")

        # create video stream
        codec = "libx264"
        pixel_format = "yuv420p"
        stream = container.add_stream(codec, rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = pixel_format
        stream.bit_rate = 1000000  # 1Mbit/s for preview quality

        for frame_img in imgs:
            if isinstance(frame_img, Image.Image):
                frame = av.VideoFrame.from_image(frame_img)
            else:
                frame = av.VideoFrame.from_ndarray(frame_img, format="rgb24")
            packets = stream.encode(frame)
            for packet in packets:
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        container.close()

        print(f"Saved video: {video_path}")


def show_datasets(
    datasets: list[BaseDataset],
    debug_mode: str,
    console_width: int,
    console_back: str,
    console_num_images: Optional[int],
    fps: int = 24,
):
    if debug_mode != "video":
        print(f"d: next dataset, q: quit")

    num_workers = max(1, os.cpu_count() - 1)
    for i, dataset in enumerate(datasets):
        print(f"Dataset [{i}]")
        batch_index = 0
        num_images_to_show = console_num_images
        k = None
        for key, batch in dataset.retrieve_latent_cache_batches(num_workers):
            print(f"bucket resolution: {key}, count: {len(batch)}")
            for j, item_info in enumerate(batch):
                item_info: ItemInfo
                print(f"{batch_index}-{j}: {item_info}")
                if debug_mode == "image":
                    k = show_image(item_info.content)
                elif debug_mode == "console":
                    k = show_console(item_info.content, console_width, console_back, console_num_images is None)
                    if num_images_to_show is not None:
                        num_images_to_show -= 1
                        if num_images_to_show == 0:
                            k = ord("d")  # next dataset
                elif debug_mode == "video":
                    save_video(item_info.content, item_info.latent_cache_path, fps)
                    k = None  # save next video

                if k == ord("q"):
                    return
                elif k == ord("d"):
                    break
            if k == ord("d"):
                break
            batch_index += 1


def encode_and_save_batch(vae: AutoencoderKLCausal3D, batch: list[ItemInfo]):
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

    # print(f"encode batch: {contents.shape}")
    with torch.no_grad():
        latent = vae.encode(contents).latent_dist.sample()
        # latent = latent * vae.config.scaling_factor

    # # debug: decode and save
    # with torch.no_grad():
    #     latent_to_decode = latent / vae.config.scaling_factor
    #     images = vae.decode(latent_to_decode, return_dict=False)[0]
    #     images = (images / 2 + 0.5).clamp(0, 1)
    #     images = images.cpu().float().numpy()
    #     images = (images * 255).astype(np.uint8)
    #     images = images.transpose(0, 2, 3, 4, 1)  # B, C, F, H, W -> B, F, H, W, C
    #     for b in range(images.shape[0]):
    #         for f in range(images.shape[1]):
    #             fln = os.path.splitext(os.path.basename(batch[b].item_key))[0]
    #             img = Image.fromarray(images[b, f])
    #             img.save(f"./logs/decode_{fln}_{b}_{f:03d}.jpg")

    for item, l in zip(batch, latent):
        # print(f"save latent cache: {item.latent_cache_path}, latent shape: {l.shape}")
        save_latent_cache(item, l)


logger = logging.getLogger(__name__) # Configura tu logger

def encode_datasets(datasets: list, encode: callable, args: object): # 'list' en lugar de list[BaseDataset] para evitar NameError si BaseDataset no est� definido aqu�
    node_name_print = "[encode_datasets function]" # Para identificar los prints
    
    # Determinar el n�mero de workers
    num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
    
    # Usar la barra de progreso de ComfyUI si se proporcion�
    use_comfy_pbar = hasattr(args, 'comfy_pbar') and args.comfy_pbar is not None
    
    if use_comfy_pbar:
        print(f"{node_name_print} Using ComfyUI ProgressBar.")
        # El ProgressBar ya deber�a estar inicializado en el nodo con el total de items/batches.
        # Aqu� solo lo actualizaremos.
    else:
        print(f"{node_name_print} ComfyUI ProgressBar not found in args. TQDM will be used if available (o no progress bar).")
        # Si tqdm estuviera disponible y quisieras usarlo como fallback:
        # from tqdm import tqdm # Import local si es fallback

    total_batches_processed_for_pbar = 0 # Para actualizar pbar por batch

    for i, dataset in enumerate(datasets):
        logger.info(f"Encoding dataset [{i}] ({type(dataset).__name__})")
        all_latent_cache_paths_for_current_dataset = [] # Renombrado para claridad

        # El m�todo dataset.retrieve_latent_cache_batches() es un generador.
        # Para usarlo con comfy_pbar, necesitamos saber cu�ntos batches va a generar
        # ANTES de iterar, o actualizar el pbar despu�s de cada batch.
        # Si no podemos saber el total de batches de antemano para este dataset,
        # el pbar que inicializamos en el nodo (con un total general) se actualizar� por cada batch global.

        # Iteramos sobre los batches que devuelve el generador del dataset
        # No usamos tqdm aqu� si tenemos comfy_pbar
        
        # Si NO puedes obtener el total de batches por adelantado para este dataset espec�fico,
        # simplemente actualiza el pbar global por cada batch que proceses.
        # El 'total' del pbar global se habr� establecido en el nodo.
        
        batch_iterator = dataset.retrieve_latent_cache_batches(num_workers)
        
        # Si quieres usar tqdm como fallback si no hay comfy_pbar:
        # if not use_comfy_pbar and 'tqdm' in globals():
        #    batch_iterator = tqdm(batch_iterator, desc=f"Dataset {i} batches")
            
        for _, batch in batch_iterator: # El primer elemento del tuple es la key del bucket, no lo usamos aqu�
            if not batch: # Si el batch est� vac�o
                continue

            all_latent_cache_paths_for_current_dataset.extend([item.latent_cache_path for item in batch])

            current_batch_to_process = batch # Usamos una copia para poder modificarla
            if args.skip_existing:
                filtered_batch_items = [item for item in current_batch_to_process if not os.path.exists(item.latent_cache_path)]
                if not filtered_batch_items:
                    if use_comfy_pbar:
                        # Aunque saltemos, si contamos items para el pbar, podr�amos querer ajustar.
                        # O si contamos batches, actualizamos que un batch se "proces�" (salt�).
                        # Si el pbar se inicializ� con el n�mero TOTAL de items, y aqu� saltamos
                        # un batch completo, debemos actualizar el pbar por el n�mero de items saltados.
                        # args.comfy_pbar.update(len(current_batch_to_process)) # Ejemplo si cuentas items
                        pass # Por ahora, no actualizamos el pbar si se salta todo el batch
                    total_batches_processed_for_pbar += 1
                    if use_comfy_pbar:
                        args.comfy_pbar.update(1) # Asumiendo que el pbar cuenta batches
                    continue
                current_batch_to_process = filtered_batch_items

            # Determinar el tama�o de sub-batch para la funci�n 'encode'
            # Tu script original lo llama 'bs', lo llamar� 'sub_batch_size' para claridad
            sub_batch_size = args.batch_size if args.batch_size is not None else len(current_batch_to_process)
            
            for j in range(0, len(current_batch_to_process), sub_batch_size):
                sub_batch = current_batch_to_process[j : j + sub_batch_size]
                if not sub_batch: # Si el sub_batch est� vac�o
                    continue
                print(f"{node_name_print} Encoding sub-batch of size {len(sub_batch)} for dataset {i}")
                encode(sub_batch) # Llama a la funci�n de encode que se le pas�

            # Actualizar la barra de progreso de ComfyUI DESPU�S de procesar un batch del dataset
            total_batches_processed_for_pbar += 1
            if use_comfy_pbar:
                args.comfy_pbar.update(1) # Asumiendo que el pbar cuenta batches

        # --- L�gica de limpieza de cach� (despu�s de procesar todos los batches de un dataset) ---
        all_latent_cache_paths_for_current_dataset = [os.path.normpath(p) for p in all_latent_cache_paths_for_current_dataset]
        all_latent_cache_paths_for_current_dataset = set(all_latent_cache_paths_for_current_dataset)

        # Asumo que dataset.get_all_latent_cache_files() existe
        if hasattr(dataset, 'get_all_latent_cache_files'):
            all_cache_files_in_dir = dataset.get_all_latent_cache_files()
            for cache_file in all_cache_files_in_dir:
                if os.path.normpath(cache_file) not in all_latent_cache_paths_for_current_dataset:
                    if args.keep_cache:
                        logger.info(f"Keep cache file not in the dataset: {cache_file}")
                    else:
                        try:
                            os.remove(cache_file)
                            logger.info(f"Removed old cache file: {cache_file}")
                        except OSError as e_remove:
                            logger.error(f"Error removing old cache file {cache_file}: {e_remove}")
        else:
            logger.warning(f"Dataset {i} does not have 'get_all_latent_cache_files' method. Skipping old cache cleanup for it.")
            
    print(f"{node_name_print} Finished processing all datasets. Total batches (from iterator) processed for pbar updates: {total_batches_processed_for_pbar}")
def main(args):
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_HUNYUAN_VIDEO)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        show_datasets(datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images)
        return

    assert args.vae is not None, "vae checkpoint is required"

    # Load VAE model: HunyuanVideo VAE model is float16
    vae_dtype = torch.float16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    vae, _, s_ratio, t_ratio = load_vae(vae_dtype=vae_dtype, device=device, vae_path=args.vae)
    vae.eval()
    logger.info(f"Loaded VAE: {vae.config}, dtype: {vae.dtype}")

    if args.vae_chunk_size is not None:
        vae.set_chunk_size_for_causal_conv_3d(args.vae_chunk_size)
        logger.info(f"Set chunk_size to {args.vae_chunk_size} for CausalConv3d in VAE")
    if args.vae_spatial_tile_sample_min_size is not None:
        vae.enable_spatial_tiling(True)
        vae.tile_sample_min_size = args.vae_spatial_tile_sample_min_size
        vae.tile_latent_min_size = args.vae_spatial_tile_sample_min_size // 8
    elif args.vae_tiling:
        vae.enable_spatial_tiling(True)

    # Encode images
    def encode(one_batch: list[ItemInfo]):
        encode_and_save_batch(vae, one_batch)

    encode_datasets(datasets, encode, args)


def setup_parser_common(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    """
    Sets up common command-line arguments.
    If a parser is provided, arguments are added to it. Otherwise, a new parser is created.
    """
    if parser is None:
        print("[cache_latents.py setup_parser_common] No parser provided, creating new one.")
        parser = argparse.ArgumentParser(description="Common Latent Caching Arguments", add_help=False) 
        # add_help=False si el parser principal (el que lo llama primero) ya tiene la ayuda general.
        # O puedes dejar que cada uno a�ada su ayuda y luego el principal usa parents.

    # ---- Common arguments ----
    parser.add_argument(
        "--dataset_config", type=str, required=True, help="path to dataset config .toml file"
    )
    parser.add_argument(
        "--vae", type=str, default=None, help="path to vae checkpoint" # Default None si es opcional
    ) 
    parser.add_argument(
        "--vae_dtype", type=str, default=None, help="data type for VAE, e.g., float16, bfloat16"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="device to use, default is cuda if available"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="batch size, override dataset config"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None, help="number of workers for dataset"
    )
    parser.add_argument(
        "--skip_existing", action="store_true", help="skip existing cache files"
    )
    parser.add_argument(
        "--keep_cache", action="store_true", help="keep cache files not in dataset"
    )
    parser.add_argument(
        "--debug_mode", type=str, default=None, choices=["image", "console", "video"], help="debug mode"
    )
    parser.add_argument(
        "--console_width", type=int, default=80, help="debug mode: console width"
    )
    parser.add_argument(
        "--console_back", type=str, default=None, help="debug mode: console background color"
    )
    parser.add_argument(
        "--console_num_images", type=int, default=None, help="debug mode: num images to show"
    )
    # ... (A�ADE TODOS tus argumentos comunes aqu� como estaban en tu script original) ...
    
    print(f"[cache_latents.py setup_parser_common] Arguments added/updated in parser: {id(parser)}")
    return parser


def hv_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--vae_tiling",
        action="store_true",
        help="enable spatial tiling for VAE, default is False. If vae_spatial_tile_sample_min_size is set, this is automatically enabled",
    )
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    parser.add_argument(
        "--vae_spatial_tile_sample_min_size", type=int, default=None, help="spatial tile sample min size for VAE, default 256"
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser_common()
    parser = hv_setup_parser(parser)

    args = parser.parse_args()
    main(args)
