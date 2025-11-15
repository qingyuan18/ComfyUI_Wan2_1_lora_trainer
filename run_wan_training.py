# -*- coding: utf-8 -*-
# File: ComfyUI/custom_nodes/musubi-tuner/run_wan_training.py
# Si el CWD del subproceso es B:\ComfyUI\ComfyUI (el directorio ra�z de ComfyUI),
# entonces 'custom_nodes' deber�a ser directamente importable.
# Si el CWD es B:\ComfyUI, entonces tambi�n.
import sys
import os
import time
import argparse
import traceback
import pathlib

# --- DEPURACI�N: Imprimir informaci�n de path y directorio ---
# Dejar esto descomentado para las proximas pruebas
#node_name_print_debug = "[WanLoraTrainerSubprocess Debug]"
#print(f"{node_name_print_debug} Script path: {os.path.abspath(__file__)}", file=sys.stderr)
#print(f"{node_name_print_debug} Script directory: {os.path.dirname(__file__)}", file=sys.stderr)
#print(f"{node_name_print_debug} Current working directory: {os.getcwd()}", file=sys.stderr)
#print(f"{node_name_print_debug} sys.path: {sys.path}", file=sys.stderr)
#print(f"{node_name_print_debug} --- End Debug Info ---", file=sys.stderr)
# --- FIN DEPURACI�N ---


# --- IMPORTACI�N Y VERIFICACI�N DE LA LIBRER�A DE ENTRENAMIENTO ---
# Esta importacion debe ser visible para el resto del script.
# Si falla, el script sale.
#print(f"{node_name_print_debug} Attempting import of training library...", file=sys.stderr)
# Preload package import so the absolute import below succeeds even without package context
try:
    _pkg_dir = os.path.dirname(__file__)
    _parent_dir = os.path.dirname(_pkg_dir)
    if _parent_dir and _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    _pkg_name = os.path.basename(_pkg_dir)
    import importlib as _importlib
    _m = _importlib.import_module(f"{_pkg_name}.wan_train_network")
    sys.modules.setdefault("wan_train_network", _m)
except Exception:
    pass

try:
    # Importaci�n ABSOLUTA del m�dulo.
    import importlib
    pkg_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(pkg_dir)
    if parent_dir and parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    wtn = importlib.import_module(f"{os.path.basename(pkg_dir)}.wan_train_network")

    # Verificar si el m�dulo importado tiene la clase y el parser necesarios
    if not hasattr(wtn, 'WanNetworkTrainer'):
        raise ImportError("WanNetworkTrainer class not found in imported module.")
    if not hasattr(wtn, 'get_full_wan_train_arg_parser'):
         # Esto es lo que tu version original lanza un raise. Convertimoslo a ImportError para consistencia.
         raise ImportError("get_full_wan_train_arg_parser function not found in imported module.")
    #print(f"{node_name_print_debug} SUCCESS: Imported training library and verified components.", file=sys.stderr)

except ImportError as e:
    print(f"ERROR: Could not import training library or find necessary components: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print(f"RESULT:ERROR:ImportError:{e}", file=sys.stdout)
    sys.stdout.flush()
    sys.exit(1)

except Exception as e:
    print(f"ERROR: Unexpected exception during training library import: {type(e).__name__}: {str(e)}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print(f"RESULT:ERROR:ImportException:{str(e)}", file=sys.stdout)
    sys.stdout.flush()
    sys.exit(1)
# --- FIN IMPORTACI�N Y VERIFICACI�N ---



def main():
    print("[run_wan_training] Script started.")

    if not (hasattr(wtn, 'get_full_wan_train_arg_parser') and callable(wtn.get_full_wan_train_arg_parser)):
        print("[run_wan_training] Error: wan_train_network.get_full_wan_train_arg_parser not found or not callable.")
        sys.exit(1)

    parser = wtn.get_full_wan_train_arg_parser()

    try:
        args = parser.parse_args()

        #Debug parse
        #print(f"[run_wan_training] Parsed arguments (before type fix): {vars(args)}")

        # --- INICIO DE LA CORRECCI�N ---
        # Asegurarse de que dataset_config sea un string si es un objeto Path
        if hasattr(args, 'dataset_config') and isinstance(args.dataset_config, pathlib.Path): # Necesitar�s `import pathlib` al inicio del script
            args.dataset_config = str(args.dataset_config)
            #print(f"[run_wan_training] Converted args.dataset_config to string: {args.dataset_config}")
        # --- CORRECCI�N PARA save_every_n_steps y save_every_n_epochs SI SON 0 ---
        if hasattr(args, "save_every_n_steps") and args.save_every_n_steps == 0:
            args.save_every_n_steps = None
            #print(f"[run_wan_training] Converted args.save_every_n_steps from 0 to None (inside run_wan_training.py).", file=sys.stderr)

        if hasattr(args, "save_every_n_epochs") and args.save_every_n_epochs == 0:
            args.save_every_n_epochs = None
            #print(f"[run_wan_training] Converted args.save_every_n_epochs from 0 to None (inside run_wan_training.py).", file=sys.stderr)
        if hasattr(args, "sample_every_n_steps") and args.sample_every_n_steps == 0:
            args.sample_every_n_steps = None

        if hasattr(args, "sample_every_n_epochs") and args.sample_every_n_epochs == 0:
            args.sample_every_n_epochs = None

        # --- FIN DE LA CORRECCI�N ---

        print(f"[run_wan_training] Parsed arguments (after type fix): {vars(args)}")

    except Exception as e_parse:
        print(f"[run_wan_training] Error during argument parsing or type conversion: {type(e_parse).__name__}: {e_parse}", file=sys.stderr)
        parser.print_help(sys.stderr) # Imprimir ayuda a stderr
        traceback.print_exc(file=sys.stderr)
        # Devolver un mensaje de error claro al proceso padre (el nodo ComfyUI)
        print(f"RESULT:ERROR:ArgParseInternalError:{e_parse}", file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1) # Salir con c�digo de error


    # Verificar que los paths obligatorios existen si se proporcionan
    required_paths_to_check = {
        "dataset_config": args.dataset_config,
        "dit": args.dit,
        "vae": args.vae,
        # t5 y clip pueden ser opcionales o ""
    }
    if hasattr(args, 't5') and args.t5: # Solo chequear si no es None o empty
        required_paths_to_check["t5"] = args.t5
    if hasattr(args, 'clip') and args.clip: # Solo chequear si no es None o empty
        required_paths_to_check["clip"] = args.clip

    for name, path_val in required_paths_to_check.items():
        if path_val and not os.path.exists(path_val):
            print(f"[run_wan_training] Error: Required path for argument --{name} does not exist: {path_val}")
            # sys.exit(1) # Considerar si esto debe ser fatal aqu� o dejar que el trainer lo maneje.
                         # Por ahora, solo advertencia, ya que el trainer puede tener l�gica m�s compleja.
                         # No, mejor salir, porque si el path no existe, el trainer fallar� igualmente.
            sys.exit(1)
        elif path_val:
            print(f"[run_wan_training] Confirmed path for --{name}: {path_val}")


    try:
        print("[run_wan_training] Initializing WanNetworkTrainer...")
        if not hasattr(wtn, 'WanNetworkTrainer'):
            print("[run_wan_training] Error: WanNetworkTrainer class not found in wan_train_network module.")
            sys.exit(1)

        trainer_instance = wtn.WanNetworkTrainer()
        print("[run_wan_training] Starting training...")
        trainer_instance.train(args) # Aqu� se llama a la funci�n de entrenamiento real
        print("[run_wan_training] Training completed successfully.")
    except SystemExit as e_sys:
        print(f"[run_wan_training] SystemExit encountered during training: code {e_sys.code}")
        # sys.exit(e_sys.code if e_sys.code is not None else 1) # Propagar el c�digo de salida
        # Re-levantar para que accelerate lo maneje si es necesario o simplemente salir
        if e_sys.code != 0 and e_sys.code is not None : # No salir si es un SystemExit(0) de argparse --help etc.
             sys.exit(e_sys.code)

    except Exception as e_train:
        print(f"[run_wan_training] Error during training execution: {e_train}")
        traceback.print_exc()
        sys.exit(1) # Indicar fallo

if __name__ == "__main__":
    main()
