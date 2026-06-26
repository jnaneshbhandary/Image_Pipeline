"""
File name: generate.py

Purpose:
Tkinter desktop GUI for queueing prompts, starting or reusing a local ComfyUI
server, submitting locked-identity workflows, and saving generated images.

How to run it:
Run `python generate.py` from the project root after running setup.py and
remember_face.py.

Prerequisites:
Python dependencies installed by setup.py, ComfyUI present under the project
root, downloaded models, copied custom node, and `identity/character.npy`.

Expected runtime:
Startup usually takes 10-60 seconds if ComfyUI is not already running. Each
image can take several minutes on an RTX 3050 4 GB GPU.
"""
from __future__ import annotations

import json
import logging
import platform
import queue
import random
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

import requests


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent
COMFYUI_DIR = PROJECT_ROOT / "ComfyUI"
IDENTITY_PATH = PROJECT_ROOT / "identity" / "character.npy"
IDENTITY_META_PATH = PROJECT_ROOT / "identity" / "character_meta.json"
OUTPUT_DIR = COMFYUI_DIR / "output"
BASE_URL = "http://127.0.0.1:8188"
IS_WINDOWS = platform.system() == "Windows"
CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0

BACKGROUND = "#1a1a1a"
PANEL = "#242424"
INPUT_BG = "#2a2a2a"
TEXT = "#e0e0e0"
WHITE = "#ffffff"
ACCENT = "#1e3a5f"
DANGER = "#5f1e1e"
ORANGE = "#5f3a1e"
WARNING_TEXT = "#e5a642"

SAMPLERS = ["dpm++2m", "dpm++2m_sde", "euler_ancestral", "dpm++3m_sde"]
SCHEDULERS = ["karras", "exponential", "sgm_uniform"]

NEGATIVE_PROMPT_CONSTANT = """(deformed iris, deformed pupils:1.4), (worst quality, low quality:1.4),
(jpeg artifacts:1.3), cgi, render, 3d, digital art, airbrushed, plastic skin,
waxy skin, glossy skin, overly smooth skin, fake skin, doll-like, mannequin,
(semi-realistic:1.3), cartoon, anime, painting, illustration, drawing, sketch,
bad anatomy, bad proportions, deformed body, mutated hands, extra fingers,
missing fingers, abnormal face, asymmetrical face, distorted face, dead eyes,
empty eyes, lifeless eyes, glazed eyes, unfocused eyes, overexposed,
underexposed, harsh lighting, oversaturated, desaturated, text, watermark,
signature, logo, multiple people, cloned face, duplicate, airbrushed face,
poreless skin, instagram filter, beauty filter, heavy makeup, cakey makeup,
motion blur, blurry, out of focus, nude, explicit, nsfw, cleavage, nipples,
topless, sexual"""


@dataclass
class PromptItem:
    prompt: str
    weight_override: float | None = None


def flush_vram(base_url: str) -> None:
    """
    Attempt to flush ComfyUI's intermediate tensor cache between prompts.
    Uses the /free endpoint if available (ComfyUI >= 2024-03).
    Falls back to a no-op workflow submission if /free is not available,
    which forces ComfyUI's execution engine to clear previous output tensors
    as a side effect of starting a new execution graph.
    This function is best-effort: it logs warnings on failure but never raises.
    """
    try:
        r = requests.post(
            f"{base_url}/free",
            json={"unload_models": False, "free_memory": True},
            timeout=5,
        )
        r.raise_for_status()
        logger.info("VRAM flush via /free succeeded.")
        return
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            logger.warning(
                "/free endpoint returned 404 - this ComfyUI version does not "
                "support it. Falling back to no-op workflow flush."
            )
        else:
            logger.warning(f"/free endpoint error: {e}. Falling back.")
    except Exception as e:
        logger.warning(f"/free endpoint unreachable: {e}. Falling back.")

    noop_workflow = {
        "1": {
            "class_type": "Note",
            "inputs": {"text": "vram_flush_noop"},
        }
    }
    try:
        requests.post(
            f"{base_url}/prompt",
            json={"prompt": noop_workflow},
            timeout=5,
        )
        logger.info("VRAM flush via no-op workflow submitted.")
    except Exception as e:
        logger.warning(
            f"No-op workflow flush also failed: {e}. "
            f"VRAM will not be explicitly cleared between prompts. "
            f"If you get OOM errors, reduce batch size."
        )


def truncate_text(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}..."


def filename_prefix(prompt: str, seed: int) -> str:
    first_chars = prompt[:30]
    filtered = "".join(ch for ch in first_chars if ch.isalnum() or ch == " ")
    filtered = re.sub(r"\s+", "_", filtered.strip().lower())
    if not filtered:
        filtered = "image"
    return f"{filtered}_{seed}"


def build_workflow(
    positive_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    ipadapter_weight: float,
) -> dict[str, Any]:
    prefix = filename_prefix(positive_prompt, seed)
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "epicrealism_v51.safetensors",
            },
        },
        "1b": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "vae-ft-mse-840000.safetensors",
            },
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": positive_prompt,
                "clip": ["1", 1],
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": NEGATIVE_PROMPT_CONSTANT,
                "clip": ["1", 1],
            },
        },
        "4": {
            "class_type": "IPAdapterModelLoader",
            "inputs": {
                "ipadapter_file": "ip-adapter-faceid-plusv2_sd15.bin",
            },
        },
        "5": {
            "class_type": "LoadEmbeddingAndInject",
            "inputs": {
                "embedding_path": "identity/character.npy",
                "ipadapter_model": ["4", 0],
                "weight": ipadapter_weight,
                "weight_type": "linear",
                "start_at": 0.0,
                "end_at": 1.0,
            },
        },
        "6": {
            "class_type": "IPAdapter",
            "inputs": {
                "model": ["1", 0],
                "ipadapter": ["5", 0],
            },
        },
        "7": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 512,
                "height": 768,
                "batch_size": 1,
            },
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["6", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["7", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
            },
        },
        "9": {
            "class_type": "LatentUpscale",
            "inputs": {
                "samples": ["8", 0],
                "upscale_method": "bislerp",
                "width": 768,
                "height": 1152,
                "crop": "disabled",
            },
        },
        "10": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["6", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["9", 0],
                "seed": seed + 1,
                "steps": 15,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 0.55,
            },
        },
        "11": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["10", 0],
                "vae": ["1b", 0],
                "tile_size": 512,
            },
        },
        "12": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["11", 0],
                "filename_prefix": prefix,
            },
        },
    }


def messages_contain_oom(messages: Any) -> bool:
    return any(
        "out of memory" in str(message).lower() or "cuda out of memory" in str(message).lower()
        for message in messages
    )


def job_has_error(job: dict[str, Any]) -> bool:
    if "error" in job:
        return True
    status = job.get("status", {})
    status_string = str(status.get("status_str", "")).lower()
    if "error" in status_string:
        return True
    messages = status.get("messages", [])
    return any("execution_error" in str(message).lower() for message in messages)


def job_error_text(job: dict[str, Any]) -> str:
    if "error" in job:
        return str(job["error"])
    status = job.get("status", {})
    messages = status.get("messages", [])
    return str(messages) if messages else str(status)


class ImageGeneratorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI Character Image Generator")
        self.root.minsize(800, 600)
        self.root.configure(bg=BACKGROUND)

        self.prompt_items: list[PromptItem] = []
        self.flagged_items: list[PromptItem] = []
        self.message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.comfyui_process: subprocess.Popen[str] | None = None
        self.process_started_by_app = False
        self.generating = False

        self.seed_var = tk.IntVar(value=-1)
        self.steps_var = tk.IntVar(value=30)
        self.cfg_var = tk.DoubleVar(value=5.0)
        self.sampler_var = tk.StringVar(value="dpm++2m")
        self.scheduler_var = tk.StringVar(value="karras")
        self.weight_var = tk.DoubleVar(value=0.8)

        self.configure_style()
        self.build_ui()
        self.refresh_identity_label()
        self.refresh_prompt_list()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.poll_queue)

    def configure_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            logger.warning("Could not apply ttk clam theme; using current theme.")
        style.configure("Dark.Horizontal.TProgressbar", troughcolor=INPUT_BG, background=ACCENT)
        style.configure("Vertical.TScrollbar", background=INPUT_BG, troughcolor=BACKGROUND)

    def build_ui(self) -> None:
        header = tk.Frame(self.root, bg=BACKGROUND)
        header.pack(fill=tk.X, padx=12, pady=(10, 6))

        title = tk.Label(
            header,
            text="AI Character Image Generator",
            bg=BACKGROUND,
            fg=WHITE,
            font=("Segoe UI", 16, "bold"),
        )
        title.pack(anchor="w")

        self.identity_label = tk.Label(
            header,
            text="",
            bg=BACKGROUND,
            fg=TEXT,
            font=("Segoe UI", 10),
        )
        self.identity_label.pack(anchor="w", pady=(2, 0))

        queue_section = tk.Frame(self.root, bg=BACKGROUND)
        queue_section.pack(fill=tk.BOTH, expand=False, padx=12, pady=(4, 8))

        queue_label = tk.Label(
            queue_section,
            text="Prompt queue",
            bg=BACKGROUND,
            fg=WHITE,
            font=("Segoe UI", 12, "bold"),
        )
        queue_label.pack(anchor="w")

        list_frame = tk.Frame(queue_section, bg=BACKGROUND)
        list_frame.pack(fill=tk.X, pady=(4, 6))

        self.prompt_listbox = tk.Listbox(
            list_frame,
            height=10,
            selectmode=tk.SINGLE,
            bg=INPUT_BG,
            fg=TEXT,
            selectbackground=ACCENT,
            selectforeground=WHITE,
            highlightthickness=1,
            highlightbackground=PANEL,
            borderwidth=0,
            font=("Consolas", 10),
        )
        self.prompt_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.prompt_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.prompt_listbox.configure(yscrollcommand=scrollbar.set)

        button_row = tk.Frame(queue_section, bg=BACKGROUND)
        button_row.pack(fill=tk.X)

        self.add_button = self.make_button(button_row, "Add prompt", self.open_add_prompt_dialog, ACCENT)
        self.remove_button = self.make_button(button_row, "Remove selected", self.remove_selected, DANGER)
        self.clear_button = self.make_button(button_row, "Clear all", self.clear_all, DANGER)
        self.flag_button = self.make_button(button_row, "Flag selected", self.flag_selected, ORANGE)
        self.retry_button = self.make_button(button_row, "Retry flagged (0)", self.retry_flagged, ORANGE)

        for button in [self.add_button, self.remove_button, self.clear_button, self.flag_button, self.retry_button]:
            button.pack(side=tk.LEFT, padx=(0, 8))

        self.build_settings_bar()

        generate_frame = tk.Frame(self.root, bg=BACKGROUND)
        generate_frame.pack(fill=tk.X, padx=12, pady=(4, 4))

        self.generate_button = self.make_button(
            generate_frame,
            "Generate",
            self.start_generation,
            ACCENT,
            font=("Segoe UI", 12, "bold"),
            pady=10,
        )
        self.generate_button.pack(fill=tk.X)

        self.status_label = tk.Label(
            self.root,
            text="Ready. 0 prompts in queue.",
            bg=BACKGROUND,
            fg=TEXT,
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.status_label.pack(fill=tk.X, padx=12, pady=(2, 4))

        self.progress_bar = ttk.Progressbar(
            self.root,
            mode="determinate",
            style="Dark.Horizontal.TProgressbar",
            maximum=1,
            value=0,
        )
        self.progress_bar.pack(fill=tk.X, padx=12, pady=(0, 8))

        log_frame = tk.Frame(self.root, bg=BACKGROUND)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.log_text = tk.Text(
            log_frame,
            height=8,
            bg=INPUT_BG,
            fg=TEXT,
            insertbackground=TEXT,
            relief=tk.FLAT,
            borderwidth=0,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Consolas", 9),
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_text.tag_configure("error", foreground="#ff7777")
        self.log_text.tag_configure("normal", foreground=TEXT)

        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

    def build_settings_bar(self) -> None:
        settings = tk.Frame(self.root, bg=BACKGROUND)
        settings.pack(fill=tk.X, padx=12, pady=(0, 8))

        self.add_spinbox(settings, "Seed", self.seed_var, -1, 2147483647, 1, width=11)
        self.add_spinbox(settings, "Steps", self.steps_var, 10, 50, 1, width=5)
        self.add_spinbox(settings, "CFG", self.cfg_var, 1.0, 15.0, 0.5, width=5)
        self.add_option(settings, "Sampler", self.sampler_var, SAMPLERS, width=13)
        self.add_option(settings, "Scheduler", self.scheduler_var, SCHEDULERS, width=11)
        self.add_spinbox(settings, "IPAdapter weight", self.weight_var, 0.1, 1.5, 0.05, width=5)

    def add_label(self, parent: tk.Frame, text: str) -> None:
        label = tk.Label(parent, text=text, bg=BACKGROUND, fg=TEXT, font=("Segoe UI", 9))
        label.pack(side=tk.LEFT, padx=(0, 4))

    def add_spinbox(
        self,
        parent: tk.Frame,
        label_text: str,
        variable: tk.Variable,
        from_value: float,
        to_value: float,
        increment: float,
        width: int,
    ) -> None:
        self.add_label(parent, label_text)
        spinbox = tk.Spinbox(
            parent,
            from_=from_value,
            to=to_value,
            increment=increment,
            textvariable=variable,
            width=width,
            bg=INPUT_BG,
            fg=TEXT,
            insertbackground=TEXT,
            buttonbackground=PANEL,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=PANEL,
        )
        spinbox.pack(side=tk.LEFT, padx=(0, 10))

    def add_option(
        self,
        parent: tk.Frame,
        label_text: str,
        variable: tk.StringVar,
        options: list[str],
        width: int,
    ) -> None:
        self.add_label(parent, label_text)
        option = tk.OptionMenu(parent, variable, *options)
        option.configure(
            bg=INPUT_BG,
            fg=TEXT,
            activebackground=ACCENT,
            activeforeground=WHITE,
            highlightthickness=0,
            relief=tk.FLAT,
            width=width,
        )
        option["menu"].configure(bg=INPUT_BG, fg=TEXT, activebackground=ACCENT, activeforeground=WHITE)
        option.pack(side=tk.LEFT, padx=(0, 10))

    def make_button(
        self,
        parent: tk.Misc,
        text: str,
        command: Any,
        background: str,
        font: tuple[str, int, str] | tuple[str, int] = ("Segoe UI", 9),
        padx: int = 10,
        pady: int = 6,
    ) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=background,
            fg=WHITE,
            activebackground=background,
            activeforeground=WHITE,
            relief=tk.FLAT,
            borderwidth=0,
            font=font,
            padx=padx,
            pady=pady,
            cursor="hand2",
        )

    def refresh_identity_label(self) -> None:
        if not IDENTITY_PATH.exists():
            self.identity_label.configure(
                text="No identity loaded - run remember_face.py first",
                fg=WARNING_TEXT,
            )
            return

        date_saved = "unknown date"
        if IDENTITY_META_PATH.exists():
            try:
                metadata = json.loads(IDENTITY_META_PATH.read_text(encoding="utf-8"))
                date_saved = str(metadata.get("saved_at", date_saved))
            except Exception as exc:
                logger.warning("Could not read identity metadata: %s", exc)
        else:
            date_saved = datetime.fromtimestamp(IDENTITY_PATH.stat().st_mtime).isoformat(timespec="seconds")

        self.identity_label.configure(
            text=f"Loaded identity: {IDENTITY_PATH.name} ({date_saved})",
            fg=TEXT,
        )

    def refresh_prompt_list(self) -> None:
        self.prompt_listbox.delete(0, tk.END)
        for index, item in enumerate(self.prompt_items, start=1):
            self.prompt_listbox.insert(tk.END, f"[{index}] {truncate_text(item.prompt, 80)}")
        self.retry_button.configure(text=f"Retry flagged ({len(self.flagged_items)})")
        if not self.generating:
            self.status_label.configure(text=f"Ready. {len(self.prompt_items)} prompts in queue.")

    def open_add_prompt_dialog(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Add prompt")
        dialog.geometry("600x200")
        dialog.resizable(False, False)
        dialog.configure(bg=BACKGROUND)
        dialog.transient(self.root)
        dialog.grab_set()

        text_box = tk.Text(
            dialog,
            height=5,
            bg=INPUT_BG,
            fg=TEXT,
            insertbackground=TEXT,
            relief=tk.FLAT,
            borderwidth=0,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
        )
        text_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(12, 8))
        text_box.focus_set()

        button_row = tk.Frame(dialog, bg=BACKGROUND)
        button_row.pack(fill=tk.X, padx=12, pady=(0, 12))

        def submit() -> None:
            prompt = text_box.get("1.0", tk.END).strip()
            if not prompt:
                messagebox.showwarning("Empty prompt", "Enter a prompt before clicking OK.")
                return
            self.prompt_items.append(PromptItem(prompt=prompt))
            self.refresh_prompt_list()
            dialog.destroy()

        ok_button = self.make_button(button_row, "OK", submit, ACCENT)
        cancel_button = self.make_button(button_row, "Cancel", dialog.destroy, DANGER)
        cancel_button.pack(side=tk.RIGHT, padx=(8, 0))
        ok_button.pack(side=tk.RIGHT)

        def on_return(event: tk.Event[Any]) -> str | None:
            shift_pressed = bool(event.state & 0x0001)
            if shift_pressed:
                return None
            submit()
            return "break"

        text_box.bind("<Return>", on_return)

    def selected_index(self) -> int | None:
        selected = self.prompt_listbox.curselection()
        if not selected:
            return None
        return int(selected[0])

    def remove_selected(self) -> None:
        index = self.selected_index()
        if index is None:
            return
        del self.prompt_items[index]
        self.refresh_prompt_list()

    def clear_all(self) -> None:
        if not self.prompt_items:
            return
        confirmed = messagebox.askyesno("Clear prompts", "Remove all prompts from the queue?")
        if confirmed:
            self.prompt_items.clear()
            self.refresh_prompt_list()

    def flag_selected(self) -> None:
        index = self.selected_index()
        if index is None:
            return
        item = self.prompt_items.pop(index)
        self.flagged_items.append(item)
        self.refresh_prompt_list()

    def retry_flagged(self) -> None:
        if not self.flagged_items:
            return
        for item in self.flagged_items:
            self.prompt_items.append(PromptItem(prompt=item.prompt, weight_override=0.9))
        self.flagged_items.clear()
        self.refresh_prompt_list()

    def collect_settings(self) -> dict[str, Any] | None:
        try:
            seed = int(self.seed_var.get())
            steps = int(self.steps_var.get())
            cfg = float(self.cfg_var.get())
            weight = float(self.weight_var.get())
        except Exception as exc:
            logger.error("Invalid settings: %s", exc)
            messagebox.showerror("Invalid settings", f"One or more settings are invalid: {exc}")
            return None

        validation_errors: list[str] = []
        if seed < -1 or seed > 2147483647:
            validation_errors.append("Seed must be between -1 and 2147483647.")
        if steps < 10 or steps > 50:
            validation_errors.append("Steps must be between 10 and 50.")
        if cfg < 1.0 or cfg > 15.0:
            validation_errors.append("CFG must be between 1.0 and 15.0.")
        if weight < 0.1 or weight > 1.5:
            validation_errors.append("IPAdapter weight must be between 0.1 and 1.5.")
        if self.sampler_var.get() not in SAMPLERS:
            validation_errors.append("Sampler selection is invalid.")
        if self.scheduler_var.get() not in SCHEDULERS:
            validation_errors.append("Scheduler selection is invalid.")

        if validation_errors:
            message = "\n".join(validation_errors)
            logger.error("Invalid settings: %s", message)
            messagebox.showerror("Invalid settings", message)
            return None

        return {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler": self.sampler_var.get(),
            "scheduler": self.scheduler_var.get(),
            "ipadapter_weight": weight,
        }

    def start_generation(self) -> None:
        if self.generating:
            return
        if not IDENTITY_PATH.exists():
            messagebox.showerror("No identity", "No identity loaded. Run remember_face.py first.")
            self.refresh_identity_label()
            return
        if not self.prompt_items:
            messagebox.showwarning("No prompts", "Add at least one prompt before generating.")
            return

        settings = self.collect_settings()
        if settings is None:
            return

        items = [PromptItem(item.prompt, item.weight_override) for item in self.prompt_items]
        self.generating = True
        self.generate_button.configure(state=tk.DISABLED)
        self.progress_bar.configure(maximum=len(items), value=0)
        self.status_label.configure(text=f"Starting. {len(items)} prompts in queue.")
        self.append_log(f"Generation started for {len(items)} prompt(s).")

        self.worker_thread = threading.Thread(
            target=self.generation_worker,
            args=(items, settings),
            daemon=True,
        )
        self.worker_thread.start()

    def generation_worker(self, items: list[PromptItem], settings: dict[str, Any]) -> None:
        saved_count = 0
        try:
            if not self.ensure_comfyui_server():
                self.message_queue.put({"type": "done", "saved": saved_count})
                return

            total = len(items)
            for index, item in enumerate(items, start=1):
                prompt = item.prompt
                short_prompt = truncate_text(prompt, 60)
                self.message_queue.put(
                    {
                        "type": "status",
                        "text": f"Generating {index} of {total} - {short_prompt}",
                    }
                )
                self.message_queue.put({"type": "progress", "value": index - 1, "total": total})

                try:
                    seed = int(settings["seed"])
                    if seed == -1:
                        seed = random.randint(0, 2147483647)
                    weight = (
                        float(item.weight_override)
                        if item.weight_override is not None
                        else float(settings["ipadapter_weight"])
                    )
                    workflow = build_workflow(
                        positive_prompt=prompt,
                        seed=seed,
                        steps=int(settings["steps"]),
                        cfg=float(settings["cfg"]),
                        sampler=str(settings["sampler"]),
                        scheduler=str(settings["scheduler"]),
                        ipadapter_weight=weight,
                    )
                    self.message_queue.put({"type": "log", "text": f"Seed used: {seed}"})
                    output_filename = self.submit_and_wait(workflow)
                    if output_filename is not None:
                        saved_count += 1
                        self.message_queue.put(
                            {
                                "type": "log",
                                "text": f"Saved: {output_filename} (seed: {seed})",
                            }
                        )
                except Exception as exc:
                    logger.exception("Generation failed for prompt: %s", prompt)
                    self.message_queue.put(
                        {
                            "type": "error",
                            "text": f"Error for prompt '{truncate_text(prompt, 60)}': {exc}",
                        }
                    )
                finally:
                    try:
                        flush_vram(BASE_URL)
                    except Exception as exc:
                        logger.warning("Unexpected VRAM flush failure ignored: %s", exc)
                    self.message_queue.put({"type": "progress", "value": index, "total": total})

            self.message_queue.put({"type": "done", "saved": saved_count})
        except Exception as exc:
            logger.exception("Unexpected worker failure.")
            self.message_queue.put({"type": "error", "text": f"Unexpected generation failure: {exc}"})
            self.message_queue.put({"type": "done", "saved": saved_count})

    def submit_and_wait(self, workflow: dict[str, Any]) -> str | None:
        response = requests.post(
            f"{BASE_URL}/prompt",
            json={"prompt": workflow},
            timeout=10,
        )
        response.raise_for_status()
        prompt_id = response.json()["prompt_id"]
        self.message_queue.put({"type": "log", "text": f"ComfyUI job ID: {prompt_id}"})

        retry_used = False
        started_at = time.monotonic()
        while True:
            if time.monotonic() - started_at > 600:
                raise TimeoutError("ComfyUI job timed out after 600 seconds.")

            history_response = requests.get(
                f"{BASE_URL}/history/{prompt_id}",
                timeout=10,
            )
            history_response.raise_for_status()
            history = history_response.json()
            job = history.get(prompt_id, {})
            status = job.get("status", {})
            messages = status.get("messages", [])

            if messages_contain_oom(messages):
                if retry_used:
                    raise RuntimeError("CUDA out of memory on reduced 640x960 retry. Skipping prompt.")
                self.message_queue.put(
                    {
                        "type": "log",
                        "text": "OOM detected on HiRes pass. Retrying at 640x960 with reduced settings...",
                    }
                )
                workflow["9"]["inputs"]["width"] = 640
                workflow["9"]["inputs"]["height"] = 960
                workflow["11"]["inputs"]["tile_size"] = 384
                retry_response = requests.post(
                    f"{BASE_URL}/prompt",
                    json={"prompt": workflow},
                    timeout=10,
                )
                retry_response.raise_for_status()
                prompt_id = retry_response.json()["prompt_id"]
                self.message_queue.put({"type": "log", "text": f"ComfyUI retry job ID: {prompt_id}"})
                retry_used = True
                started_at = time.monotonic()
                continue

            if job and job_has_error(job):
                raise RuntimeError(f"ComfyUI job failed: {job_error_text(job)}")

            if prompt_id in history and job.get("outputs"):
                outputs = job["outputs"]
                image_outputs = outputs.get("12", {}).get("images", [])
                if not image_outputs:
                    raise RuntimeError("ComfyUI completed but SaveImage output was missing.")
                image_info = image_outputs[0]
                filename = image_info["filename"]
                subfolder = str(image_info.get("subfolder", "")).strip()
                output_path = OUTPUT_DIR / subfolder / filename if subfolder else OUTPUT_DIR / filename
                logger.info("Saved image: %s", output_path)
                return str(output_path)

            time.sleep(2)

    def ensure_comfyui_server(self) -> bool:
        if self.server_reachable():
            self.message_queue.put({"type": "log", "text": "ComfyUI server already running."})
            self.process_started_by_app = False
            return True

        if not (COMFYUI_DIR / "main.py").exists():
            self.message_queue.put(
                {
                    "type": "error",
                    "text": f"ComfyUI not found at {COMFYUI_DIR}. Run setup.py first.",
                }
            )
            return False

        self.message_queue.put({"type": "status", "text": "Starting ComfyUI server..."})
        self.message_queue.put({"type": "log", "text": "Starting ComfyUI server..."})
        try:
            self.comfyui_process = subprocess.Popen(
                [sys.executable, "main.py", "--port", "8188"],
                cwd=str(COMFYUI_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=CREATE_NO_WINDOW,
            )
            self.process_started_by_app = True
        except Exception as exc:
            self.message_queue.put({"type": "error", "text": f"Could not start ComfyUI: {exc}"})
            return False

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.comfyui_process.poll() is not None:
                stdout, stderr = self.comfyui_process.communicate(timeout=5)
                self.message_queue.put(
                    {
                        "type": "error",
                        "text": f"ComfyUI exited during startup.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}",
                    }
                )
                return False
            if self.server_reachable():
                self.message_queue.put({"type": "log", "text": "ComfyUI server is ready."})
                self.start_pipe_drain_threads(self.comfyui_process)
                return True
            time.sleep(2)

        stderr_output = ""
        if self.comfyui_process.poll() is None:
            self.comfyui_process.terminate()
            try:
                _, stderr_output = self.comfyui_process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                self.comfyui_process.kill()
                _, stderr_output = self.comfyui_process.communicate(timeout=5)
        self.message_queue.put(
            {
                "type": "error",
                "text": f"ComfyUI server did not start within 60 seconds. {stderr_output}",
            }
        )
        return False

    def start_pipe_drain_threads(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is not None:
            threading.Thread(
                target=self.drain_pipe,
                args=(process.stdout, logging.INFO),
                daemon=True,
            ).start()
        if process.stderr is not None:
            threading.Thread(
                target=self.drain_pipe,
                args=(process.stderr, logging.ERROR),
                daemon=True,
            ).start()

    def drain_pipe(self, pipe: Any, level: int) -> None:
        try:
            for line in iter(pipe.readline, ""):
                cleaned = line.strip()
                if cleaned:
                    logger.log(level, "[ComfyUI] %s", cleaned)
        except Exception as exc:
            logger.warning("Stopped reading ComfyUI process pipe: %s", exc)

    def server_reachable(self) -> bool:
        try:
            response = requests.get(f"{BASE_URL}/system_stats", timeout=2)
            return response.status_code == 200
        except Exception as exc:
            logger.debug("ComfyUI server reachability probe failed: %s", exc)
            return False

    def poll_queue(self) -> None:
        try:
            while True:
                message = self.message_queue.get_nowait()
                message_type = message.get("type")
                if message_type == "status":
                    self.status_label.configure(text=str(message.get("text", "")))
                elif message_type == "progress":
                    total = int(message.get("total", 1))
                    value = int(message.get("value", 0))
                    self.progress_bar.configure(maximum=max(total, 1), value=value)
                elif message_type == "log":
                    self.append_log(str(message.get("text", "")))
                elif message_type == "error":
                    self.append_log(str(message.get("text", "")), tag="error")
                elif message_type == "done":
                    saved = int(message.get("saved", 0))
                    self.generating = False
                    self.generate_button.configure(state=tk.NORMAL)
                    self.status_label.configure(text=f"Done. {saved} images saved to output/.")
                    self.append_log(f"Generation done. {saved} image(s) saved.")
                    self.refresh_identity_label()
                    self.refresh_prompt_list()
        except queue.Empty:
            logger.debug("UI message queue is empty.")
            pass
        self.root.after(100, self.poll_queue)

    def append_log(self, text: str, tag: str = "normal") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {text}\n", tag)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        if tag == "error":
            logger.error("%s", text)
        else:
            logger.info("%s", text)

    def on_close(self) -> None:
        if self.process_started_by_app and self.comfyui_process is not None:
            if self.comfyui_process.poll() is None:
                logger.info("Terminating ComfyUI subprocess started by generate.py.")
                self.comfyui_process.terminate()
                try:
                    self.comfyui_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("ComfyUI did not exit after terminate; killing process.")
                    self.comfyui_process.kill()
                    self.comfyui_process.wait(timeout=5)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ImageGeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
