#!/usr/bin/env python3

"""
everything u are  4
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import re
import time
import yaml
import toml
import shutil
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config, save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType
from state_manager import get_state, set_state, clear_state
from datetime import datetime


def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path
def merge_model_config(default_config: dict, model_config: dict) -> dict:
    merged = {}

    if isinstance(default_config, dict):
        merged.update(default_config)

    if isinstance(model_config, dict):
        merged.update(model_config)

    return merged if merged else None

def count_images_in_directory(directory_path: str) -> int:
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    count = 0
    
    try:
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}", flush=True)
            return 0
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.startswith('.'):
                    continue
                
                _, ext = os.path.splitext(file.lower())
                if ext in image_extensions:
                    count += 1
    except Exception as e:
        print(f"Error counting images in directory: {e}", flush=True)
        return 0
    
    return count



def get_config_for_model(lrs_config: dict, model_name: str) -> dict:
    if not isinstance(lrs_config, dict):
        return None

    data = lrs_config.get("data")
    default_config = lrs_config.get("default", {})

    if isinstance(data, dict) and model_name in data:
        return merge_model_config(default_config, data.get(model_name))

    if default_config:
        return default_config

    return None

def load_lrs_config(model_type: str, is_style: bool) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "lrs")

    if model_type == "flux":
        config_file = os.path.join(config_dir, "flux.json")
    elif is_style:
        config_file = os.path.join(config_dir, "style_config.json")
    else:
        config_file = os.path.join(config_dir, "person_config.json")
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load LRS config from {config_file}: {e}", flush=True)
        return None


def create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word: str | None = None):
    """Get the training data directory"""
    train_data_dir = train_paths.get_image_training_images_dir(task_id)

    """Create the diffusion config file"""
    config_template_path, is_style = train_paths.get_image_training_config_template_path(model_type, train_data_dir)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        with open(config_template_path, "r") as file:
            config = yaml.safe_load(file)
        if 'config' in config and 'process' in config['config']:
            for process in config['config']['process']:
                if 'model' in process:
                    process['model']['name_or_path'] = model_path
                    if 'training_folder' in process:
                        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        process['training_folder'] = output_dir
                
                if 'datasets' in process:
                    for dataset in process['datasets']:
                        dataset['folder_path'] = train_data_dir

                if trigger_word:
                    process['trigger_word'] = trigger_word
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
        save_config(config, config_path)
        print(f"Created ai-toolkit config at {config_path}", flush=True)
        return config_path
    else:
        with open(config_template_path, "r") as file:
            config = toml.load(file)

        dataset_size = 0
        if os.path.exists(train_data_dir):
            dataset_size = count_images_in_directory(train_data_dir)
            if dataset_size > 0:
                print(f"Counted {dataset_size} images in training directory", flush=True)

        size_config_loaded = False
        lrs_config = load_lrs_config(model_type, is_style)

        if lrs_config:
            model_hash = hash_model(model_name)
            lrs_settings = get_config_for_model(lrs_config, model_hash)

            if lrs_settings:
                if model_type == "flux":
                    print(f"Applying model-specific config for Flux model", flush=True)
                    for key, value in lrs_settings.items():
                        config[key] = value
                else:
                    size_key = None
                    if 1 <= dataset_size <= 10:
                        size_key = "xs"
                    elif 11 <= dataset_size <= 20:
                        size_key = "s"
                    elif 21 <= dataset_size <= 30:
                        size_key = "m"
                    elif 31 <= dataset_size <= 50:
                        size_key = "l"
                    elif 51 <= dataset_size <= 1000:
                        size_key = "xl"
                    
                    if size_key and size_key in lrs_settings:
                        print(f"Applying model-specific config for size '{size_key}'", flush=True)
                        for key, value in lrs_settings[size_key].items():
                            config[key] = value
                        size_config_loaded = True
                    else:
                        print(f"Warning: No size configuration '{size_key}' found for model '{model_name}'.", flush=True)
            else:
                print(f"Warning: No LRS configuration found for model '{model_name}'", flush=True)
        else:
            print("Warning: Could not load LRS configuration, using default values", flush=True)

        network_config_person = {
            "stabilityai/stable-diffusion-xl-base-1.0": 235,
            "Lykon/dreamshaper-xl-1-0": 235,
            "Lykon/art-diffusion-xl-0.9": 235,
            "SG161222/RealVisXL_V4.0": 467,
            "stablediffusionapi/protovision-xl-v6.6": 235,
            "stablediffusionapi/omnium-sdxl": 235,
            "GraydientPlatformAPI/realism-engine2-xl": 235,
            "GraydientPlatformAPI/albedobase2-xl": 467,
            "KBlueLeaf/Kohaku-XL-Zeta": 235,
            "John6666/hassaku-xl-illustrious-v10style-sdxl": 228,
            "John6666/nova-anime-xl-pony-v5-sdxl": 235,
            "cagliostrolab/animagine-xl-4.0": 699,
            "dataautogpt3/CALAMITY": 235,
            "dataautogpt3/ProteusSigma": 235,
            "dataautogpt3/ProteusV0.5": 467,
            "dataautogpt3/TempestV0.1": 456,
            "ehristoforu/Visionix-alpha": 235,
            "femboysLover/RealisticStockPhoto-fp16": 467,
            "fluently/Fluently-XL-Final": 228,
            "mann-e/Mann-E_Dreams": 456,
            "misri/leosamsHelloworldXL_helloworldXL70": 235,
            "misri/zavychromaxl_v90": 235,
            "openart-custom/DynaVisionXL": 228,
            "recoilme/colorfulxl": 228,
            "zenless-lab/sdxl-aam-xl-anime-mix": 456,
            "zenless-lab/sdxl-anima-pencil-xl-v5": 228,
            "zenless-lab/sdxl-anything-xl": 228,
            "zenless-lab/sdxl-blue-pencil-xl-v7": 467,
            "Corcelio/mobius": 228,
            "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
            "OnomaAIResearch/Illustrious-xl-early-release-v0": 228
        }

        network_config_style = {
            "stabilityai/stable-diffusion-xl-base-1.0": 235,
            "Lykon/dreamshaper-xl-1-0": 235,
            "Lykon/art-diffusion-xl-0.9": 235,
            "SG161222/RealVisXL_V4.0": 235,
            "stablediffusionapi/protovision-xl-v6.6": 235,
            "stablediffusionapi/omnium-sdxl": 235,
            "GraydientPlatformAPI/realism-engine2-xl": 235,
            "GraydientPlatformAPI/albedobase2-xl": 235,
            "KBlueLeaf/Kohaku-XL-Zeta": 235,
            "John6666/hassaku-xl-illustrious-v10style-sdxl": 235,
            "John6666/nova-anime-xl-pony-v5-sdxl": 235,
            "cagliostrolab/animagine-xl-4.0": 235,
            "dataautogpt3/CALAMITY": 235,
            "dataautogpt3/ProteusSigma": 235,
            "dataautogpt3/ProteusV0.5": 235,
            "dataautogpt3/TempestV0.1": 228,
            "ehristoforu/Visionix-alpha": 235,
            "femboysLover/RealisticStockPhoto-fp16": 235,
            "fluently/Fluently-XL-Final": 235,
            "mann-e/Mann-E_Dreams": 235,
            "misri/leosamsHelloworldXL_helloworldXL70": 235,
            "misri/zavychromaxl_v90": 235,
            "openart-custom/DynaVisionXL": 235,
            "recoilme/colorfulxl": 235,
            "zenless-lab/sdxl-aam-xl-anime-mix": 235,
            "zenless-lab/sdxl-anima-pencil-xl-v5": 235,
            "zenless-lab/sdxl-anything-xl": 235,
            "zenless-lab/sdxl-blue-pencil-xl-v7": 235,
            "Corcelio/mobius": 235,
            "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
            "OnomaAIResearch/Illustrious-xl-early-release-v0": 235
        }

        config_mapping = {
            228: {
                "network_dim": 32,
                "network_alpha": 32,
                "network_args": []
            },
            235: {
                "network_dim": 32,
                "network_alpha": 32,
                "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
            },
            456: {
                "network_dim": 64,
                "network_alpha": 64,
                "network_args": []
            },
            467: {
                "network_dim": 64,
                "network_alpha": 64,
                "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
            },
            699: {
                "network_dim": 96,
                "network_alpha": 96,
                "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
            },
        }

        config["pretrained_model_name_or_path"] = model_path
        config["train_data_dir"] = train_data_dir
        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        config["output_dir"] = output_dir

        if model_type == "sdxl":
            if is_style:
                network_config = config_mapping[network_config_style[model_name]]
            else:
                network_config = config_mapping[network_config_person[model_name]]

            config["network_dim"] = network_config["network_dim"]
            config["network_alpha"] = network_config["network_alpha"]
            config["network_args"] = network_config["network_args"]


        # Old size config search removed as requested
        if dataset_size > 0 and not size_config_loaded:
             print(f"Warning: No size-specific configuration (xs/s/m/l/xl) found for model '{model_name}' with {dataset_size} images. Using model defaults.", flush=True)
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
        save_config_toml(config, config_path)
        print(f"config is {config}", flush=True)
        print(f"Created config at {config_path}", flush=True)
        return config_path


# Error detection constants
OOM_ERROR = "torch.OutOfMemoryError: CUDA out of memory"
OOM_ERROR_ALT = "CUDA out of memory"
OOM_ERROR_ALT2 = "RuntimeError: CUDA out of memory"


def get_error_type(log_path: str) -> str | None:
    """Detect error type from log file."""
    if not os.path.exists(log_path):
        return None
    
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        if OOM_ERROR in text or OOM_ERROR_ALT in text or OOM_ERROR_ALT2 in text:
            return OOM_ERROR
    except Exception as e:
        print(f"Error reading log file: {e}", flush=True)
    
    return None


def reduce_batch_size_in_config(config_path: str, model_type: str) -> bool:
    """Reduce batch size in config file to handle OOM errors. Returns True if modified."""
    try:
        if model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]:
            # For YAML configs (AI Toolkit)
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            
            modified = False
            if 'config' in config and 'process' in config['config']:
                for process in config['config']['process']:
                    if 'training' in process:
                        training = process['training']
                        if 'batch_size' in training:
                            current_batch = training['batch_size']
                            if current_batch > 1:
                                new_batch = max(1, current_batch // 2)
                                training['batch_size'] = new_batch
                                modified = True
                                print(f"Reduced batch size from {current_batch} to {new_batch} in config", flush=True)
            
            if modified:
                with open(config_path, "w") as file:
                    yaml.dump(config, file, default_flow_style=False)
            
            return modified
        else:
            # For TOML configs (SDXL/Flux)
            with open(config_path, "r") as file:
                config = toml.load(file)
            
            modified = False
            # Common batch size keys in diffusion training
            batch_keys = ['train_batch_size', 'batch_size', 'gradient_accumulation_steps']
            
            for key in batch_keys:
                if key in config:
                    current_value = config[key]
                    if isinstance(current_value, int) and current_value > 1:
                        if key == 'gradient_accumulation_steps':
                            # For gradient accumulation, we can reduce it
                            new_value = max(1, current_value // 2)
                        else:
                            # For batch size, reduce it
                            new_value = max(1, current_value // 2)
                        
                        config[key] = new_value
                        modified = True
                        print(f"Reduced {key} from {current_value} to {new_value} in config", flush=True)
            
            if modified:
                save_config_toml(config, config_path)
            
            return modified
    except Exception as e:
        print(f"Error modifying config for batch size reduction: {e}", flush=True)
        return False


def run_cmd_with_log(cmd: list[str], log_file_path: str, env_vars: dict = None, append: bool = False):
    """Run command with logging to both console and file."""
    print(f"Running command: {' '.join(cmd)}", flush=True)
    
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file_path) if os.path.dirname(log_file_path) else "."
    os.makedirs(log_dir, exist_ok=True)
    
    mode = "a" if append else "w"
    with open(log_file_path, mode, encoding="utf-8") as log_file:
        # Prepare environment variables
        process_env = os.environ.copy()
        if env_vars:
            process_env.update(env_vars)

        # Run the command, capturing stdout and stderr
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )

        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        # Wait for the process to complete
        return_code = process.wait()

        # Log the return code
        log_file.write(f"\nProcess completed with return code: {return_code}\n")
        log_file.flush()
        
        return return_code


def run_training(model_type: str, config_path: str, task_id: str, retries: int = 3):
    """Run training with retry logic and OOM error handling."""
    print(f"Starting training with config: {config_path}", flush=True)
    
    # Initialize state for this task
    state = get_state(task_id)
    if not state:
        state = {
            "mode": "initial",
            "task_id": task_id,
            "model_type": model_type,
            "config_path": config_path,
            "attempts": [],
            "oom_errors": 0,
            "batch_size_reductions": 0,
            "start_time": datetime.now().isoformat(),
        }
        set_state(state, task_id)
    else:
        print(f"Resuming training from previous state for task {task_id}", flush=True)
    
    # Create log directory
    log_dir = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_{task_id}.log")

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    for attempt in range(retries):
        print(
            f"************* Training attempt {attempt + 1}/{retries} for task {task_id} *************",
            flush=True,
        )
        
        # On retry after first attempt, check for OOM and reduce batch size
        if attempt > 0:
            # Load state to check previous attempts
            state = get_state(task_id)
            previous_attempt = state.get("attempts", [])[-1] if state.get("attempts") else None
            
            if os.path.exists(log_path) and previous_attempt:
                error_type = get_error_type(log_path)
                if error_type == OOM_ERROR:
                    print(f"OOM error detected (attempt {previous_attempt.get('attempt_number', attempt)}), attempting to reduce batch size...", flush=True)
                    state["oom_errors"] = state.get("oom_errors", 0) + 1
                    # Save OOM error count even if batch size reduction fails
                    set_state(state, task_id)
                    
                    if reduce_batch_size_in_config(config_path, model_type):
                        state["batch_size_reductions"] = state.get("batch_size_reductions", 0) + 1
                        print(f"Config modified for retry attempt {attempt + 1} (batch size reduction #{state['batch_size_reductions']})", flush=True)
                        set_state(state, task_id)  # Save state after batch size reduction
                    else:
                        print(f"Could not reduce batch size further or config doesn't support it", flush=True)
                else:
                    print(f"Previous attempt failed with error type: {error_type}", flush=True)
                    # Update state with error type even if not OOM
                    if previous_attempt:
                        previous_attempt["error_type"] = error_type
                        set_state(state, task_id)
            
            # Add separator for new attempt in log
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"STARTING TRAINING ATTEMPT {attempt + 1}\n")
                f.write(f"{'='*80}\n\n")
        
        # Build training command
        if is_ai_toolkit:
            training_command = [
                "python3",
                "/app/ai-toolkit/run.py",
                config_path
            ]
        else:
            if model_type == "sdxl":
                training_command = [
                    "accelerate", "launch",
                    "--dynamo_backend", "no",
                    "--dynamo_mode", "default",
                    "--mixed_precision", "bf16",
                    "--num_processes", "1",
                    "--num_machines", "1",
                    "--num_cpu_threads_per_process", "2",
                    f"/app/sd-script/{model_type}_train_network.py",
                    "--config_file", config_path
                ]
            elif model_type == "flux":
                training_command = [
                    "accelerate", "launch",
                    "--dynamo_backend", "no",
                    "--dynamo_mode", "default",
                    "--mixed_precision", "bf16",
                    "--num_processes", "1",
                    "--num_machines", "1",
                    "--num_cpu_threads_per_process", "2",
                    f"/app/sd-scripts/{model_type}_train_network.py",
                    "--config_file", config_path
                ]
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        # Get fresh state for current attempt (may have been updated in retry logic above)
        state = get_state(task_id)
        attempt_info = {
            "attempt_number": attempt + 1,
            "start_time": datetime.now().isoformat(),
            "config_path": config_path,
            "command": " ".join(training_command),
        }
        # Save attempt start info to state before training
        state["attempts"] = state.get("attempts", []) + [attempt_info.copy()]
        state["mode"] = "training"
        set_state(state, task_id)
        
        try:
            print("Starting training subprocess...\n", flush=True)
            return_code = run_cmd_with_log(training_command, log_path, append=(attempt > 0))
            
            attempt_info["end_time"] = datetime.now().isoformat()
            attempt_info["return_code"] = return_code
            attempt_info["error_type"] = get_error_type(log_path) if return_code != 0 else None
            
            # Get fresh state and update the last attempt
            state = get_state(task_id)
            # Ensure state structure exists (shouldn't happen, but safety check)
            if not state:
                state = {
                    "mode": "initial",
                    "task_id": task_id,
                    "model_type": model_type,
                    "config_path": config_path,
                    "attempts": [],
                    "oom_errors": 0,
                    "batch_size_reductions": 0,
                }
            
            if state.get("attempts"):
                # Update the last attempt (which we just added before training)
                state["attempts"][-1].update(attempt_info)
            else:
                # Fallback: add attempt if somehow missing
                state["attempts"] = [attempt_info]
            
            if return_code == 0:
                print("Training subprocess completed successfully.", flush=True)
                attempt_info["success"] = True
                state["attempts"][-1]["success"] = True
                state["mode"] = "success"
                state["end_time"] = datetime.now().isoformat()
                set_state(state, task_id)
                return True
            else:
                print(f"Training subprocess failed with return code: {return_code}", flush=True)
                attempt_info["success"] = False
                state["attempts"][-1]["success"] = False
                
                # Check if it's the last attempt
                if attempt < retries - 1:
                    state["mode"] = "retrying"
                    print(f"Retrying in 5 seconds...", flush=True)
                    set_state(state, task_id)
                    time.sleep(5)
                else:
                    state["mode"] = "failed"
                    state["end_time"] = datetime.now().isoformat()
                    print(f"All {retries} attempts failed.", flush=True)
                    set_state(state, task_id)
                    return False

        except Exception as e:
            print(f"Training subprocess error: {e}", flush=True)
            # Get fresh state and update
            state = get_state(task_id)
            # Ensure state structure exists
            if not state:
                state = {
                    "mode": "initial",
                    "task_id": task_id,
                    "model_type": model_type,
                    "config_path": config_path,
                    "attempts": [],
                    "oom_errors": 0,
                    "batch_size_reductions": 0,
                }
            
            attempt_info["end_time"] = datetime.now().isoformat()
            attempt_info["success"] = False
            attempt_info["exception"] = str(e)
            if state.get("attempts"):
                # Update the last attempt if it exists
                state["attempts"][-1].update(attempt_info)
            else:
                # Add as new attempt if list is empty
                state["attempts"] = [attempt_info]
            
            if attempt < retries - 1:
                state["mode"] = "retrying"
                print(f"Retrying in 5 seconds...", flush=True)
                set_state(state, task_id)
                time.sleep(5)
            else:
                state["mode"] = "failed"
                state["end_time"] = datetime.now().isoformat()
                print(f"All {retries} attempts failed.", flush=True)
                set_state(state, task_id)
                raise RuntimeError(f"Training failed after {retries} attempts: {e}")
    
    # Final state update
    state = get_state(task_id)
    if state.get("mode") != "success":
        state["mode"] = "failed"
        state["end_time"] = datetime.now().isoformat()
        set_state(state, task_id)
    
    return False

def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed


def print_state_summary(task_id: str):
    """Print a summary of the training state."""
    state = get_state(task_id)
    if not state:
        print("No state found for this task.", flush=True)
        return
    
    print("\n" + "="*80, flush=True)
    print("TRAINING STATE SUMMARY", flush=True)
    print("="*80, flush=True)
    print(f"Task ID: {state.get('task_id', 'N/A')}", flush=True)
    print(f"Model Type: {state.get('model_type', 'N/A')}", flush=True)
    print(f"Mode: {state.get('mode', 'N/A')}", flush=True)
    print(f"Start Time: {state.get('start_time', 'N/A')}", flush=True)
    print(f"End Time: {state.get('end_time', 'N/A')}", flush=True)
    print(f"OOM Errors: {state.get('oom_errors', 0)}", flush=True)
    print(f"Batch Size Reductions: {state.get('batch_size_reductions', 0)}", flush=True)
    print(f"Total Attempts: {len(state.get('attempts', []))}", flush=True)
    
    attempts = state.get('attempts', [])
    if attempts:
        print("\nAttempt History:", flush=True)
        for i, attempt in enumerate(attempts, 1):
            print(f"  Attempt {i}:", flush=True)
            print(f"    Start: {attempt.get('start_time', 'N/A')}", flush=True)
            print(f"    End: {attempt.get('end_time', 'N/A')}", flush=True)
            print(f"    Success: {attempt.get('success', False)}", flush=True)
            print(f"    Return Code: {attempt.get('return_code', 'N/A')}", flush=True)
            if attempt.get('error_type'):
                print(f"    Error Type: {attempt.get('error_type')}", flush=True)
    
    print("="*80 + "\n", flush=True) 

async def main():
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux", "qwen-image", "z-image"], help="Model type")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--trigger-word", help="Trigger word for the training")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    parser.add_argument("--retries", type=int, default=3, help="Number of retry attempts on failure (default: 3)")
    parser.add_argument("--clear-state", action="store_true", help="Clear previous state for this task before starting")
    args = parser.parse_args()
    
    # Clear state if requested
    if args.clear_state:
        clear_state(args.task_id)
        print(f"Cleared previous state for task {args.task_id}", flush=True)

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)

    model_path = train_paths.get_image_base_model_path(args.model)

    print("Preparing dataset...", flush=True)

    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    config_path = create_config(
        args.task_id,
        model_path,
        args.model,
        args.model_type,
        args.expected_repo_name,
        args.trigger_word,
    )

    # Run training with retry logic
    success = run_training(
        model_type=args.model_type,
        config_path=config_path,
        task_id=args.task_id,
        retries=args.retries
    )
    
    # Print state summary
    print_state_summary(args.task_id)
    
    if not success:
        print(f"Training failed for task {args.task_id} after all retry attempts", flush=True)
        sys.exit(1)
    
    print(f"Training completed successfully for task {args.task_id}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())