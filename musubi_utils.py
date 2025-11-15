import os
import sys

# Try to import ComfyUI's folder_paths. If unavailable (e.g., subprocess context),
# walk up the directory tree to find ComfyUI root and add it to sys.path. As a
# last resort, provide a minimal stub so imports don't crash in training subprocesses.
try:
    import folder_paths  # Provided by ComfyUI at repo root
except Exception:
    _search_dir = os.path.dirname(__file__)
    _found = False
    for _ in range(6):  # up to 6 levels just in case
        _candidate = os.path.join(_search_dir, "folder_paths.py")
        if os.path.isfile(_candidate):
            if _search_dir not in sys.path:
                sys.path.insert(0, _search_dir)
            try:
                import folder_paths  # type: ignore
                _found = True
                break
            except Exception:
                pass
        _parent = os.path.dirname(_search_dir)
        if _parent == _search_dir:
            break
        _search_dir = _parent
    if not _found:
        class _FolderPathsStub:
            @staticmethod
            def get_folder_paths(name):
                return []
            @staticmethod
            def get_filename_list(name):
                return []
        folder_paths = _FolderPathsStub()  # type: ignore

def get_model_files(folder_name, extensions):
    try:
        folders = folder_paths.get_folder_paths(folder_name)
        all_files = []
        for folder in folders:
            if os.path.isdir(folder):
                for item_name in os.listdir(folder):
                    if os.path.isfile(os.path.join(folder, item_name)) and any(item_name.lower().endswith(ext) for ext in extensions):
                        all_files.append(item_name)
        return sorted(list(set(all_files)))
    except Exception as e:
        print(f"[MusubiTuner Utils] ERROR escaneando '{folder_name}': {e}")
        return []

MODEL_EXTENSIONS = ['.pth', '.safetensors']

models_combo = [""] + get_model_files("diffusion_models", MODEL_EXTENSIONS)
vaes_combo = [""] + get_model_files("vae", MODEL_EXTENSIONS)
encoders_combo = [""] + get_model_files("text_encoders", MODEL_EXTENSIONS)
try:
    clip_vision_files = ["None"] + folder_paths.get_filename_list("clip_vision")
except:
    clip_vision_files = ["None"]