"""
File name: setup.py

Purpose:
Bootstrap a Windows ComfyUI environment for the locked-identity image pipeline.
It installs Python packages, clones ComfyUI, downloads required models, installs
the custom nodes, and runs a CPU-only ComfyUI health check.

How to run it:
Run `python setup.py` from the project root.

Prerequisites:
Python 3.10 or 3.11, NVIDIA drivers for CUDA use, internet access, and Git for
Windows available on PATH.

Expected runtime:
Typically 20-60 minutes on a clean machine, depending on download speed.
"""
from __future__ import annotations

import importlib
import logging
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent
COMFYUI_DIR = PROJECT_ROOT / "ComfyUI"
CUSTOM_NODE_SOURCE = PROJECT_ROOT / "custom_nodes" / "load_embedding_inject.py"
CUSTOM_NODE_TARGET = COMFYUI_DIR / "custom_nodes" / "load_embedding_inject.py"
IPADAPTER_PLUS_DIR = COMFYUI_DIR / "custom_nodes" / "ComfyUI_IPAdapter_plus"
IPADAPTER_PLUS_COMMIT = "4e1a0fd"
IS_WINDOWS = platform.system() == "Windows"
CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0

SUMMARY: list[tuple[str, str]] = []


def record(step: str, status: str) -> None:
    SUMMARY.append((step, status))


def print_summary() -> None:
    if not SUMMARY:
        return
    print("")
    print("+----------------------------------------------+--------+")
    print("| Step                                         | Status |")
    print("+----------------------------------------------+--------+")
    for step, status in SUMMARY:
        print(f"| {step[:44]:<44} | {status[:6]:<6} |")
    print("+----------------------------------------------+--------+")


def fail(step: str, message: str, exc: Exception | None = None) -> None:
    record(step, "FAIL")
    if exc is not None:
        logger.error("%s: %s", message, exc)
    else:
        logger.error("%s", message)
    print_summary()
    sys.exit(1)


def run_command(command: list[str], step: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    try:
        logger.info("Running: %s", " ".join(command))
        return subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=CREATE_NO_WINDOW,
        )
    except FileNotFoundError as exc:
        fail(step, f"Command not found: {command[0]}", exc)
    except subprocess.CalledProcessError as exc:
        logger.error("Command stdout:\n%s", exc.stdout.strip())
        logger.error("Command stderr:\n%s", exc.stderr.strip())
        fail(step, f"Command failed: {' '.join(command)}", exc)
    raise RuntimeError("unreachable")


def run_command_allow_failure(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    logger.info("Running: %s", " ".join(command))
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=CREATE_NO_WINDOW,
    )


def check_python_version() -> None:
    if sys.version_info < (3, 10):
        fail(
            "Python version",
            f"Python >= 3.10 is required. Detected {sys.version.split()[0]}.",
        )
    logger.info("Python version OK: %s", sys.version.split()[0])
    record("Python version", "OK")


def check_torch_cuda(label: str, warn_only: bool = True) -> bool | None:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        logger.info("Torch is not installed yet; skipping CUDA check.")
        record(label, "SKIP")
        return None

    try:
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = getattr(torch.version, "cuda", None)
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("CUDA detected. CUDA version: %s. GPU: %s", cuda_version, gpu_name)
            record(label, "OK")
            return True
        message = (
            "CUDA is not currently available to torch. Setup will continue; "
            "verify NVIDIA drivers if generation later runs on CPU."
        )
        if warn_only:
            logger.warning(message)
            record(label, "WARN")
            return False
        fail(label, message)
    except Exception as exc:
        if warn_only:
            logger.warning("Could not complete CUDA check: %s", exc)
            record(label, "WARN")
            return False
        fail(label, "Could not complete CUDA check.", exc)
    return False


def install_python_dependencies() -> None:
    packages: list[list[str]] = [
        [
            "torch==2.2.2",
            "torchvision==0.17.2",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
        ["insightface==0.7.3"],
        ["onnxruntime-gpu==1.17.3"],
        ["opencv-python==4.9.0.80"],
        ["numpy==1.26.4"],
        ["Pillow==10.3.0"],
        ["requests==2.31.0"],
        ["tqdm==4.66.2"],
        ["scipy==1.13.0"],
    ]

    for package_args in packages:
        step = f"pip install {' '.join(package_args[:2]) if len(package_args) > 1 else package_args[0]}"
        command = [sys.executable, "-m", "pip", "install", *package_args]
        run_command(command, step)
        logger.info("Installed successfully: %s", " ".join(package_args))

    record("Python dependencies", "OK")
    importlib.invalidate_caches()


def clone_comfyui() -> None:
    if COMFYUI_DIR.exists() and (COMFYUI_DIR / "main.py").exists():
        logger.info("ComfyUI already present, skipping clone.")
        record("Clone ComfyUI", "SKIP")
    else:
        if shutil.which("git") is None:
            fail(
                "Clone ComfyUI",
                "Git for Windows is required. Install it from https://git-scm.com/download/win and rerun setup.py.",
            )
        if COMFYUI_DIR.exists() and not (COMFYUI_DIR / "main.py").exists():
            fail(
                "Clone ComfyUI",
                f"{COMFYUI_DIR} exists but does not look like a ComfyUI checkout. "
                "Move or remove it, then rerun setup.py.",
            )
        run_command(
            ["git", "clone", "https://github.com/comfyanonymous/ComfyUI.git", str(COMFYUI_DIR)],
            "Clone ComfyUI",
        )
        record("Clone ComfyUI", "OK")

    requirements = COMFYUI_DIR / "requirements.txt"
    if not requirements.exists():
        fail("ComfyUI requirements", f"Missing ComfyUI requirements file: {requirements}")
    run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements)], "ComfyUI requirements")
    record("ComfyUI requirements", "OK")


def create_folder_structure() -> None:
    directories = [
        COMFYUI_DIR / "models" / "checkpoints",
        COMFYUI_DIR / "models" / "ipadapter",
        COMFYUI_DIR / "models" / "clip_vision",
        COMFYUI_DIR / "models" / "loras",
        COMFYUI_DIR / "models" / "upscale_models",
        COMFYUI_DIR / "models" / "vae",
        COMFYUI_DIR / "input",
        COMFYUI_DIR / "output",
        PROJECT_ROOT / "identity",
        PROJECT_ROOT / "custom_nodes",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    logger.info("Folder structure is ready.")
    record("Folder structure", "OK")


def _import_download_dependencies() -> tuple[Any, Any]:
    try:
        requests = importlib.import_module("requests")
        tqdm_module = importlib.import_module("tqdm")
        return requests, tqdm_module.tqdm
    except ImportError as exc:
        fail("Download dependencies", "requests and tqdm must be installed before model downloads.", exc)
    raise RuntimeError("unreachable")


def download_file(url: str, target_path: Path) -> None:
    requests, tqdm = _import_download_dependencies()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        if target_path.stat().st_size > 1024 * 1024:
            logger.info("Already downloaded: %s", target_path.name)
            return
        logger.warning("Existing file is too small and will be re-downloaded: %s", target_path.name)
        target_path.unlink()

    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    backoff_seconds = 2
    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
        resume_pos = tmp_path.stat().st_size if tmp_path.exists() else 0
        headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}

        try:
            with requests.get(url, stream=True, timeout=30, headers=headers) as response:
                if resume_pos > 0 and response.status_code == 200:
                    logger.warning("Server ignored resume request for %s; restarting download.", target_path.name)
                    tmp_path.unlink(missing_ok=True)
                    resume_pos = 0
                elif response.status_code not in (200, 206):
                    response.raise_for_status()

                content_length = int(response.headers.get("content-length", "0") or "0")
                total_size = resume_pos + content_length if content_length else None
                total_mb = (total_size or 0) / (1024 * 1024)
                mode = "ab" if resume_pos > 0 and response.status_code == 206 else "wb"

                with tmp_path.open(mode) as output_file:
                    with tqdm(
                        total=total_size,
                        initial=resume_pos if mode == "ab" else 0,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"{target_path.name} ({total_mb:.1f} MB)" if total_size else target_path.name,
                    ) as progress:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                output_file.write(chunk)
                                progress.update(len(chunk))

            if tmp_path.stat().st_size <= 1024 * 1024:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(f"Downloaded file is smaller than 1 MB: {target_path.name}")

            tmp_path.replace(target_path)
            logger.info("Downloaded: %s", target_path.name)
            return

        except Exception as exc:
            if attempt == max_attempts:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(f"Failed to download {target_path.name} after {max_attempts} attempts.") from exc
            logger.warning(
                "Download attempt %s/%s failed for %s: %s. Retrying in %s seconds.",
                attempt,
                max_attempts,
                target_path.name,
                exc,
                backoff_seconds,
            )
            time.sleep(backoff_seconds)
            backoff_seconds *= 2


def download_models() -> None:
    models = [
        (
            "https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1_fp16-no-ema.safetensors",
            COMFYUI_DIR / "models" / "checkpoints" / "epicrealism_v51.safetensors",
        ),
        (
            "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",
            COMFYUI_DIR / "models" / "vae" / "vae-ft-mse-840000.safetensors",
        ),
        (
            "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin",
            COMFYUI_DIR / "models" / "ipadapter" / "ip-adapter-faceid-plusv2_sd15.bin",
        ),
        (
            "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
            COMFYUI_DIR / "models" / "clip_vision" / "clip_vision_vit_h.safetensors",
        ),
        (
            "https://huggingface.co/Kim2091/4xUltraSharp/resolve/main/4x-UltraSharp.pth",
            COMFYUI_DIR / "models" / "upscale_models" / "4x-UltraSharp.pth",
        ),
    ]

    try:
        for url, target_path in models:
            download_file(url, target_path)
    except Exception as exc:
        fail("Model downloads", "A model download failed.", exc)

    vae_path = COMFYUI_DIR / "models" / "vae" / "vae-ft-mse-840000.safetensors"
    if not vae_path.exists() or vae_path.stat().st_size <= 1024 * 1024:
        fail(
            "VAE verification",
            "Mandatory SD 1.5 VAE is missing or corrupt. "
            "The noVAE checkpoint cannot decode images without it.",
        )

    record("Model downloads", "OK")
    record("Mandatory VAE verification", "OK")


def install_ipadapter_plus() -> None:
    if shutil.which("git") is None:
        fail(
            "Install ComfyUI_IPAdapter_plus",
            "Git for Windows is required. Install it from https://git-scm.com/download/win and rerun setup.py.",
        )

    if not IPADAPTER_PLUS_DIR.exists():
        run_command(
            [
                "git",
                "clone",
                "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git",
                str(IPADAPTER_PLUS_DIR),
            ],
            "Install ComfyUI_IPAdapter_plus",
        )
    else:
        logger.info("ComfyUI_IPAdapter_plus already present.")

    current_commit = ""
    if (IPADAPTER_PLUS_DIR / ".git").exists():
        result = run_command(["git", "rev-parse", "--short", "HEAD"], "Check IPAdapter commit", cwd=IPADAPTER_PLUS_DIR)
        current_commit = result.stdout.strip()

    if current_commit.startswith(IPADAPTER_PLUS_COMMIT):
        logger.info("ComfyUI_IPAdapter_plus already pinned to %s.", IPADAPTER_PLUS_COMMIT)
        record("ComfyUI_IPAdapter_plus clone", "SKIP")
    else:
        fetch_result = run_command_allow_failure(["git", "fetch", "--all"], cwd=IPADAPTER_PLUS_DIR)
        checkout_result = run_command_allow_failure(
            ["git", "checkout", IPADAPTER_PLUS_COMMIT],
            cwd=IPADAPTER_PLUS_DIR,
        )
        pin_ok = fetch_result.returncode == 0 and checkout_result.returncode == 0

        if pin_ok:
            logger.info("Pinned ComfyUI_IPAdapter_plus to %s.", IPADAPTER_PLUS_COMMIT)
            record("ComfyUI_IPAdapter_plus pin", "OK")
        else:
            logger.warning("git fetch stdout: %s", fetch_result.stdout.strip())
            logger.warning("git fetch stderr: %s", fetch_result.stderr.strip())
            logger.warning("git checkout stdout: %s", checkout_result.stdout.strip())
            logger.warning("git checkout stderr: %s", checkout_result.stderr.strip())
            logger.warning(
                "WARNING: Could not pin ComfyUI_IPAdapter_plus to known-good commit 4e1a0fd. "
                "The injection node may fail if the repo has changed its internal API. "
                "Check custom_nodes/load_embedding_inject.py if face identity does not lock."
            )
            record("ComfyUI_IPAdapter_plus pin", "WARN")

    requirements = IPADAPTER_PLUS_DIR / "requirements.txt"
    if requirements.exists():
        run_command(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
            "ComfyUI_IPAdapter_plus requirements",
        )
        record("ComfyUI_IPAdapter_plus requirements", "OK")
    else:
        logger.info("No ComfyUI_IPAdapter_plus requirements.txt found; skipping.")
        record("ComfyUI_IPAdapter_plus requirements", "SKIP")


def install_project_custom_node() -> None:
    if not CUSTOM_NODE_SOURCE.exists():
        logger.warning(
            "Project custom node source does not exist yet: %s. It will be copied on the next run of setup.py.",
            CUSTOM_NODE_SOURCE,
        )
        record("Project custom node copy", "WARN")
        return

    CUSTOM_NODE_TARGET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CUSTOM_NODE_SOURCE, CUSTOM_NODE_TARGET)
    logger.info("Copied project custom node to ComfyUI: %s", CUSTOM_NODE_TARGET)
    record("Project custom node copy", "OK")


def health_check_comfyui() -> None:
    requests, _ = _import_download_dependencies()
    command = [sys.executable, "main.py", "--port", "8188", "--cpu"]
    process: subprocess.Popen[str] | None = None

    try:
        process = subprocess.Popen(
            command,
            cwd=str(COMFYUI_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=CREATE_NO_WINDOW,
        )

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            try:
                response = requests.get("http://127.0.0.1:8188/system_stats", timeout=2)
                if response.status_code == 200:
                    logger.info("ComfyUI health check passed.")
                    record("ComfyUI health check", "OK")
                    return
            except Exception as exc:
                logger.debug("ComfyUI not ready yet: %s", exc)
            time.sleep(2)

        stderr_output = ""
        if process.poll() is None:
            process.terminate()
            try:
                _, stderr_output = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                _, stderr_output = process.communicate(timeout=5)
        else:
            _, stderr_output = process.communicate(timeout=5)
        fail("ComfyUI health check", f"ComfyUI did not respond within 30 seconds.\n{stderr_output.strip()}")

    except FileNotFoundError as exc:
        fail("ComfyUI health check", "Could not start ComfyUI health check.", exc)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main() -> None:
    check_python_version()
    check_torch_cuda("CUDA check before install")
    install_python_dependencies()
    check_torch_cuda("CUDA check after install")
    clone_comfyui()
    create_folder_structure()
    download_models()
    install_ipadapter_plus()
    install_project_custom_node()
    health_check_comfyui()
    print_summary()
    print("Setup complete. You may now run remember_face.py.")


if __name__ == "__main__":
    main()
