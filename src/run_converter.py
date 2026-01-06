import sys
import os
import shutil
import argparse
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

BANNER = """
===========================================
      EASY Model Converter (by classic)
===========================================
"""

def setup_dlls(gpu_type):
    print(f"[INFO] Setting up DLLs for GPU type: {gpu_type}")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dll_path = None

    if gpu_type == "118":
        dll_path = os.path.join(base_dir, "dlls_11.8")
    elif gpu_type in ["126", "128"]:
        dll_path = os.path.join(base_dir, "dlls_12.x")
    elif gpu_type == "amd":
        print("[INFO] AMD Mode: No NVIDIA DLLs required.")
        return

    if dll_path and os.path.isdir(dll_path):
        os.environ["PATH"] = dll_path + os.pathsep + os.environ["PATH"]
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(dll_path)
            except Exception:
                pass
        print(f"[INFO] Loaded TensorRT DLLs from: {os.path.basename(dll_path)}")
    else:
        print(f"[WARN] DLL folder not found: {dll_path}")
        print("      TensorRT conversion might fail if not installed on system.")

def check_requirements():
    missing = []
    
    try:
        import torch
        print(f"[INFO] System PyTorch found: {torch.__version__}")
    except ImportError:
        missing.append("torch")

    try:
        import ultralytics
        print(f"[INFO] System Ultralytics found: {ultralytics.__version__}")
    except ImportError:
        missing.append("ultralytics")

    try:
        import onnxruntime
        print(f"[INFO] ONNX Runtime found: {onnxruntime.__version__}")
    except ImportError:
        missing.append("onnxruntime")
        
    if missing:
        print("\n" + "!"*40)
        print(" [CRITICAL ERROR] MISSING LIBRARIES")
        print("!"*40)
        print(f"This converter requires the following installed on your system:")
        for m in missing:
            print(f" - {m}")
        print("\nPlease run this command in your terminal:")
        print(f"pip install {' '.join(missing)}")
        print("\nPress Enter to exit...")
        input()
        sys.exit(1)

def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select YOLO Model (.pt or .onnx)",
        filetypes=[("YOLO Models", "*.pt *.onnx"), ("PyTorch Model", "*.pt"), ("ONNX Model", "*.onnx")]
    )
    return file_path

def onnx_to_engine(onnx_file_path, use_half):
    print(f"\n[INFO] Starting ONNX to TensorRT (.engine) conversion...")
    
    onnx2engine = None
    try:
        from ultralytics.utils.export.engine import onnx2engine
    except (ImportError, ModuleNotFoundError):
        try:
            from ultralytics.utils.export import onnx2engine
        except (ImportError, ModuleNotFoundError):
            print("[ERROR] Could not find 'onnx2engine'. Update Ultralytics: pip install -U ultralytics")
            raise Exception("Ultralytics version incompatible.")

    output_file = Path(onnx_file_path).with_suffix('.engine')
    
    mode_str = "FP16 (Half)" if use_half else "FP32 (Full)"
    print(f"[INFO] Converting in STATIC mode (dynamic=False) with precision: {mode_str}...")
    
    onnx2engine(
        onnx_file=onnx_file_path, 
        engine_file=str(output_file), 
        half=use_half,
        dynamic=False,
    )
    
    return str(output_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", required=True, help="GPU Type config")
    args = parser.parse_args()

    os.system('cls' if os.name == 'nt' else 'clear')
    print(BANNER)
    
    setup_dlls(args.type)
    check_requirements()
    from ultralytics import YOLO

    print("\n[STEP 1] Please select your **.pt** or **.onnx** model file...")
    model_path = select_file()
    if not model_path:
        print("No file selected. Exiting.")
        return

    print(f"Selected: {model_path}")
    model_ext = Path(model_path).suffix.lower()

    print("\n[STEP 2] Choose Output Format:")
    
    export_options = {}
    if model_ext == ".pt":
        print(" 1. ONNX (.onnx)")
        print(" 2. TensorRT (.engine) - (Nvidia Only)")
        export_options = {"1": "onnx", "2": "engine"}
    elif model_ext == ".onnx":
        print(" 1. TensorRT (.engine) - (Nvidia Only)")
        export_options = {"1": "engine"}
    else:
        print(f"[ERROR] Unsupported model type: {model_ext}")
        input("\nPress Enter to close...")
        return
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    export_format = export_options.get(choice)
    
    if not export_format:
        print("[ERROR] Invalid choice. Exiting.")
        input("\nPress Enter to close...")
        return
    
    if export_format == "engine" and args.type == "amd":
        print("\n[WARN] AMD cannot export to TensorRT. Exiting.")
        input("\nPress Enter to close...")
        return

    use_half = False
    if export_format == "engine":
        print("\n[STEP 3] Choose Precision:")
        print(" 1. FP16 (Recommended - Faster, Less Memory)")
        print(" 2. FP32 (Full Precision - Higher Accuracy)")
        p_choice = input("\nEnter choice (1 or 2): ").strip()
        if p_choice == "1":
            use_half = True
        else:
            use_half = False

    try:
        exported_path = None
        
        if model_ext == ".pt":
            print(f"\n[INFO] Loading PyTorch Model...")
            model = YOLO(model_path)
            
            mode_str = ""
            if export_format == "engine":
                mode_str = "FP16" if use_half else "FP32"
                
            print(f"[INFO] Starting Export to {export_format.upper()} {mode_str}...")
            device = 0 if args.type != "amd" else "cpu"
            
            exported_path = model.export(
                format=export_format, 
                simplify=True, 
                device=device, 
                half=use_half
            )

        elif model_ext == ".onnx" and export_format == "engine":
            exported_path = onnx_to_engine(model_path, use_half)
            
        if exported_path:
            print("\n" + "="*30)
            print("       CONVERSION SUCCESS")
            print("="*30)
            print(f"Saved to: {exported_path}")
            print("Made with <3 by classic")
        else:
            raise Exception("Conversion failed to return a path.")
            
    except Exception as e:
        print(f"\n[ERROR] Conversion Failed.")
        print(f"Details: {e}")
        messagebox.showerror("Error", f"Conversion failed:\n{str(e)}")
if __name__ == "__main__":
    main()
