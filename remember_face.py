"""
File name: remember_face.py

Purpose:
Open a Tkinter GUI for selecting reference photos, extracting InsightFace
ArcFace embeddings on CPU, reviewing quality warnings, and saving one averaged
L2-normalised identity vector for the fictional character.

How to run it:
Run `python remember_face.py` from the project root.

Prerequisites:
Run `python setup.py` first so InsightFace, OpenCV, NumPy, and Pillow are
installed. The first run may download the Buffalo_L InsightFace model.

Expected runtime:
Usually 1-5 minutes for 5-7 photos, plus the initial Buffalo_L model download.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image
from PIL import ImageDraw
from PIL import ImageTk


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent
IDENTITY_DIR = PROJECT_ROOT / "identity"
IDENTITY_PATH = IDENTITY_DIR / "character.npy"
METADATA_PATH = IDENTITY_DIR / "character_meta.json"

BACKGROUND = "#1a1a1a"
PANEL = "#242424"
INPUT_BG = "#2a2a2a"
TEXT = "#e0e0e0"
WHITE = "#ffffff"
ACCENT = "#1e3a5f"
WARNING = "#b88a1e"
POSE = "#b85f1e"
DANGER = "#5f1e1e"
OK = "#1f7a3a"
CARD_WIDTH = 200
CARD_HEIGHT = 260
THUMB_SIZE = 150


def shorten_filename(path: Path, limit: int = 20) -> str:
    name = path.name
    if len(name) <= limit:
        return name
    return f"{name[: limit - 3]}..."


def clamp_int(value: float, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(round(value))))


def face_bbox(face: Any, width: int, height: int, padding: int = 0) -> tuple[int, int, int, int]:
    bbox = np.asarray(face.bbox, dtype=np.float32)
    x1 = clamp_int(bbox[0] - padding, 0, width - 1)
    y1 = clamp_int(bbox[1] - padding, 0, height - 1)
    x2 = clamp_int(bbox[2] + padding, x1 + 1, width)
    y2 = clamp_int(bbox[3] + padding, y1 + 1, height)
    return x1, y1, x2, y2


def calculate_quality(image_rgb: np.ndarray, face: Any) -> tuple[float, float, float]:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = face_bbox(face, width, height)
    crop_rgb = image_rgb[y1:y2, x1:x2]
    if crop_rgb.size == 0:
        sharpness = 0.0
    else:
        grey_crop = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(grey_crop, cv2.CV_64F).var())

    yaw = 0.0
    kps = getattr(face, "kps", None)
    if kps is not None and len(kps) >= 3:
        keypoints = np.asarray(kps, dtype=np.float32)
        left_eye = keypoints[0]
        right_eye = keypoints[1]
        nose_tip = keypoints[2]
        eye_mid_x = float((left_eye[0] + right_eye[0]) / 2.0)
        face_width = max(float(np.asarray(face.bbox)[2] - np.asarray(face.bbox)[0]), 1.0)
        yaw = abs(float((nose_tip[0] - eye_mid_x) / face_width) * 180.0)

    quality_score = max(0.0, sharpness - (yaw * 2.0))
    return quality_score, sharpness, yaw


def normalise_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    normalised = normalise_embeddings(embeddings.astype(np.float32))
    return normalised @ normalised.T


def average_pairwise_similarity(matrix: np.ndarray) -> float:
    if matrix.shape[0] <= 1:
        return 1.0
    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return float(np.mean(upper)) if upper.size else 1.0


class FaceIdentityExtractor:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Face Identity Extractor")
        self.root.geometry("900x700")
        self.root.resizable(False, False)
        self.root.configure(bg=BACKGROUND)

        self.message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.results: list[dict[str, Any]] = []
        self.image_refs: list[ImageTk.PhotoImage] = []
        self.worker_thread: threading.Thread | None = None
        self.progress_bar: ttk.Progressbar | None = None
        self.current_file_label: tk.Label | None = None

        self.configure_style()
        self.show_file_selection()
        self.root.after(100, self.poll_queue)

    def configure_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            logger.warning("Could not apply ttk clam theme; using current theme.")
        style.configure("Dark.Horizontal.TProgressbar", troughcolor=INPUT_BG, background=ACCENT)

    def clear_window(self) -> None:
        for child in self.root.winfo_children():
            child.destroy()

    def make_button(
        self,
        parent: tk.Misc,
        text: str,
        command: Any,
        background: str = ACCENT,
        font: tuple[str, int, str] | tuple[str, int] = ("Segoe UI", 11),
        padx: int = 12,
        pady: int = 8,
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

    def show_file_selection(self) -> None:
        self.clear_window()
        container = tk.Frame(self.root, bg=BACKGROUND)
        container.pack(expand=True)
        button = self.make_button(
            container,
            "Select Reference Photos",
            self.select_reference_photos,
            background=ACCENT,
            font=("Segoe UI", 14),
            padx=24,
            pady=12,
        )
        button.pack()

    def select_reference_photos(self) -> None:
        selected = filedialog.askopenfilenames(
            title="Select 5-7 reference photos of your character",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp"), ("All files", "*.*")],
        )
        paths = [Path(path) for path in selected]
        if len(paths) < 2 or len(paths) > 10:
            messagebox.showwarning("Invalid selection", "Select between 2 and 10 reference photos.")
            return
        self.start_extraction(paths)

    def start_extraction(self, paths: list[Path]) -> None:
        self.clear_window()

        container = tk.Frame(self.root, bg=BACKGROUND)
        container.pack(expand=True, fill=tk.BOTH, padx=40, pady=40)

        title = tk.Label(
            container,
            text="Extracting face identity...",
            bg=BACKGROUND,
            fg=WHITE,
            font=("Segoe UI", 16, "bold"),
        )
        title.pack(pady=(180, 20))

        self.progress_bar = ttk.Progressbar(
            container,
            mode="indeterminate",
            style="Dark.Horizontal.TProgressbar",
            length=500,
        )
        self.progress_bar.pack(pady=10)
        self.progress_bar.start(12)

        self.current_file_label = tk.Label(
            container,
            text="Preparing InsightFace...",
            bg=BACKGROUND,
            fg=TEXT,
            font=("Segoe UI", 11),
        )
        self.current_file_label.pack(pady=10)

        self.worker_thread = threading.Thread(target=self.extract_worker, args=(paths,), daemon=True)
        self.worker_thread.start()

    def extract_worker(self, paths: list[Path]) -> None:
        try:
            buffalo_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
            if not buffalo_dir.exists():
                logger.info("Downloading InsightFace Buffalo_L model (~300 MB)...")
                self.message_queue.put(
                    {
                        "type": "status",
                        "text": "Downloading InsightFace Buffalo_L model (~300 MB)...",
                    }
                )

            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))

            results: list[dict[str, Any]] = []
            for path in paths:
                self.message_queue.put({"type": "status", "text": f"Processing {path.name}"})
                result = self.process_image(path, app)
                results.append(result)

            self.message_queue.put({"type": "extraction_done", "results": results})
        except Exception as exc:
            logger.exception("Face extraction failed.")
            self.message_queue.put({"type": "fatal_error", "text": f"Face extraction failed: {exc}"})

    def process_image(self, path: Path, app: FaceAnalysis) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": path,
            "image_rgb": None,
            "faces": [],
            "face": None,
            "selected_face_index": None,
            "embedding": None,
            "quality_score": 0.0,
            "sharpness": 0.0,
            "yaw": 0.0,
            "status": "no_face",
            "checkbox_var": None,
        }

        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            logger.warning("OpenCV could not read image: %s", path)
            return result

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result["image_rgb"] = image_rgb
        faces = app.get(image_rgb)
        result["faces"] = faces

        if not faces:
            result["status"] = "no_face"
            return result

        selected_face = None
        selected_index: int | None = None
        was_multiple_faces = len(faces) > 1
        if was_multiple_faces:
            result["status"] = "multiple_faces"
            event = threading.Event()
            holder: dict[str, int | None] = {"index": None}
            self.message_queue.put(
                {
                    "type": "select_face",
                    "path": path,
                    "image_rgb": image_rgb,
                    "faces": faces,
                    "event": event,
                    "holder": holder,
                }
            )
            event.wait()
            selected_index = holder.get("index")
            if selected_index is None:
                logger.warning("No face selected for multi-face image: %s", path)
                return result
            selected_face = faces[selected_index]
        else:
            selected_index = 0
            selected_face = faces[0]

        quality_score, sharpness, yaw = calculate_quality(image_rgb, selected_face)
        embedding = np.asarray(selected_face.embedding, dtype=np.float32)

        result["face"] = selected_face
        result["selected_face_index"] = selected_index
        result["embedding"] = embedding
        result["quality_score"] = quality_score
        result["sharpness"] = sharpness
        result["yaw"] = yaw

        if was_multiple_faces:
            result["status"] = "ok"
        elif sharpness < 80:
            result["status"] = "blurry"
        elif yaw > 35:
            result["status"] = "extreme_pose"
        else:
            result["status"] = "ok"

        return result

    def poll_queue(self) -> None:
        try:
            while True:
                message = self.message_queue.get_nowait()
                message_type = message.get("type")
                if message_type == "status":
                    if self.current_file_label is not None:
                        self.current_file_label.configure(text=str(message.get("text", "")))
                elif message_type == "select_face":
                    self.show_multi_face_window(message)
                elif message_type == "extraction_done":
                    if self.progress_bar is not None:
                        self.progress_bar.stop()
                    self.results = list(message["results"])
                    self.show_results()
                elif message_type == "fatal_error":
                    if self.progress_bar is not None:
                        self.progress_bar.stop()
                    messagebox.showerror("Extraction failed", str(message.get("text", "Unknown error")))
                    self.show_file_selection()
        except queue.Empty:
            logger.debug("UI message queue is empty.")
            pass
        self.root.after(100, self.poll_queue)

    def show_multi_face_window(self, message: dict[str, Any]) -> None:
        path: Path = message["path"]
        image_rgb: np.ndarray = message["image_rgb"]
        faces: list[Any] = message["faces"]
        event: threading.Event = message["event"]
        holder: dict[str, int | None] = message["holder"]

        window = tk.Toplevel(self.root)
        window.title("Select face")
        window.geometry("400x300")
        window.resizable(False, False)
        window.configure(bg=BACKGROUND)
        window.transient(self.root)
        window.grab_set()

        label = tk.Label(
            window,
            text=f"Select the character face in {shorten_filename(path, 32)}",
            bg=BACKGROUND,
            fg=WHITE,
            font=("Segoe UI", 10, "bold"),
        )
        label.pack(pady=(12, 8))

        face_frame = tk.Frame(window, bg=BACKGROUND)
        face_frame.pack(expand=True, fill=tk.BOTH, padx=12, pady=8)

        local_refs: list[ImageTk.PhotoImage] = []

        def choose(index: int) -> None:
            holder["index"] = index
            event.set()
            window.destroy()

        for index, face in enumerate(faces):
            thumbnail = self.build_thumbnail(image_rgb, face, size=100)
            photo = ImageTk.PhotoImage(thumbnail)
            local_refs.append(photo)
            button = tk.Button(
                face_frame,
                image=photo,
                text=str(index + 1),
                compound=tk.TOP,
                command=lambda idx=index: choose(idx),
                bg=PANEL,
                fg=WHITE,
                activebackground=INPUT_BG,
                activeforeground=WHITE,
                relief=tk.FLAT,
                borderwidth=1,
                padx=6,
                pady=6,
                cursor="hand2",
            )
            button.grid(row=index // 3, column=index % 3, padx=8, pady=8)

        window._image_refs = local_refs  # type: ignore[attr-defined]

        def on_close() -> None:
            holder["index"] = None
            event.set()
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", on_close)

    def build_thumbnail(self, image_rgb: np.ndarray | None, face: Any | None, size: int = THUMB_SIZE) -> Image.Image:
        if image_rgb is None:
            return Image.new("RGB", (size, size), INPUT_BG)

        height, width = image_rgb.shape[:2]
        if face is not None:
            crop_x1, crop_y1, crop_x2, crop_y2 = face_bbox(face, width, height, padding=20)
            crop = image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                crop = image_rgb
                crop_x1 = 0
                crop_y1 = 0
            image = Image.fromarray(crop).convert("RGB")
            original_width, original_height = image.size
            image = image.resize((size, size), Image.Resampling.LANCZOS)

            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = face_bbox(face, width, height)
            scale_x = size / max(original_width, 1)
            scale_y = size / max(original_height, 1)
            draw = ImageDraw.Draw(image)
            draw.rectangle(
                [
                    (bbox_x1 - crop_x1) * scale_x,
                    (bbox_y1 - crop_y1) * scale_y,
                    (bbox_x2 - crop_x1) * scale_x,
                    (bbox_y2 - crop_y1) * scale_y,
                ],
                outline="#00ff66",
                width=3,
            )
            return image

        image = Image.fromarray(image_rgb).convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), INPUT_BG)
        offset = ((size - image.width) // 2, (size - image.height) // 2)
        canvas.paste(image, offset)
        return canvas

    def show_results(self) -> None:
        self.clear_window()
        self.image_refs.clear()

        title = tk.Label(
            self.root,
            text="Review extracted identity",
            bg=BACKGROUND,
            fg=WHITE,
            font=("Segoe UI", 16, "bold"),
        )
        title.pack(pady=(12, 6))

        outer = tk.Frame(self.root, bg=BACKGROUND)
        outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        canvas = tk.Canvas(outer, bg=BACKGROUND, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        content = tk.Frame(canvas, bg=BACKGROUND)
        content_window = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def on_frame_configure(_event: tk.Event[Any]) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event: tk.Event[Any]) -> None:
            canvas.itemconfigure(content_window, width=event.width)

        content.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        grid = tk.Frame(content, bg=BACKGROUND)
        grid.pack(fill=tk.X, padx=8, pady=8)

        for index, result in enumerate(self.results):
            self.create_result_card(grid, result, index)

        self.create_similarity_section(content)

        save_frame = tk.Frame(self.root, bg=BACKGROUND)
        save_frame.pack(fill=tk.X, padx=12, pady=(0, 12))
        save_button = self.make_button(
            save_frame,
            "Save Identity",
            self.save_identity,
            background=ACCENT,
            font=("Segoe UI", 12, "bold"),
            pady=10,
        )
        save_button.pack(fill=tk.X)

    def create_result_card(self, parent: tk.Frame, result: dict[str, Any], index: int) -> None:
        card = tk.Frame(parent, bg=PANEL, width=CARD_WIDTH, height=CARD_HEIGHT, highlightthickness=1)
        card.grid(row=index // 4, column=index % 4, padx=8, pady=8, sticky="n")
        card.grid_propagate(False)

        thumbnail = self.build_thumbnail(result.get("image_rgb"), result.get("face"))
        photo = ImageTk.PhotoImage(thumbnail)
        self.image_refs.append(photo)

        image_label = tk.Label(card, image=photo, bg=PANEL)
        image_label.pack(pady=(8, 4))

        filename = tk.Label(
            card,
            text=shorten_filename(result["path"]),
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 9),
        )
        filename.pack()

        status_text, status_color = self.status_badge(result["status"])
        badge = tk.Label(
            card,
            text=status_text,
            bg=status_color,
            fg=WHITE,
            font=("Segoe UI", 8, "bold"),
            padx=6,
            pady=2,
        )
        badge.pack(pady=4)

        sharpness = tk.Label(
            card,
            text=f"Sharpness: {float(result.get('sharpness') or 0.0):.1f}",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 9),
        )
        sharpness.pack()

        checked_by_default = result["status"] == "ok"
        variable = tk.BooleanVar(value=checked_by_default)
        result["checkbox_var"] = variable
        checkbox = tk.Checkbutton(
            card,
            text="Use",
            variable=variable,
            bg=PANEL,
            fg=TEXT,
            activebackground=PANEL,
            activeforeground=TEXT,
            selectcolor=INPUT_BG,
            font=("Segoe UI", 9),
        )
        checkbox.pack(pady=2)

    def status_badge(self, status: str) -> tuple[str, str]:
        if status == "ok":
            return "OK", OK
        if status == "blurry":
            return "BLURRY", WARNING
        if status == "extreme_pose":
            return "POSE", POSE
        if status == "multiple_faces":
            return "MULTI", WARNING
        return "NO FACE", DANGER

    def create_similarity_section(self, parent: tk.Frame) -> None:
        section = tk.Frame(parent, bg=BACKGROUND)
        section.pack(fill=tk.X, padx=8, pady=(10, 16))

        title = tk.Label(
            section,
            text="Similarity matrix",
            bg=BACKGROUND,
            fg=WHITE,
            font=("Segoe UI", 12, "bold"),
        )
        title.pack(anchor="w", pady=(0, 6))

        valid_results = [
            result
            for result in self.results
            if result.get("embedding") is not None and result.get("status") in {"ok", "multiple_faces"}
        ]
        if len(valid_results) < 2:
            label = tk.Label(
                section,
                text="Not enough OK images to compute pairwise similarity.",
                bg=BACKGROUND,
                fg=WARNING,
                font=("Segoe UI", 10),
            )
            label.pack(anchor="w")
            return

        embeddings = np.stack([np.asarray(result["embedding"], dtype=np.float32) for result in valid_results])
        matrix = cosine_matrix(embeddings)

        matrix_frame = tk.Frame(section, bg=BACKGROUND)
        matrix_frame.pack(anchor="w")

        tk.Label(matrix_frame, text="", bg=BACKGROUND, fg=TEXT, width=10).grid(row=0, column=0, padx=1, pady=1)
        for col, result in enumerate(valid_results, start=1):
            label = tk.Label(
                matrix_frame,
                text=shorten_filename(result["path"], 8),
                bg=BACKGROUND,
                fg=TEXT,
                width=8,
                font=("Segoe UI", 8),
            )
            label.grid(row=0, column=col, padx=1, pady=1)

        for row, result in enumerate(valid_results, start=1):
            row_label = tk.Label(
                matrix_frame,
                text=shorten_filename(result["path"], 8),
                bg=BACKGROUND,
                fg=TEXT,
                width=10,
                anchor="e",
                font=("Segoe UI", 8),
            )
            row_label.grid(row=row, column=0, padx=1, pady=1)
            for col in range(len(valid_results)):
                value = float(matrix[row - 1, col])
                cell = tk.Label(
                    matrix_frame,
                    text=f"{value:.2f}",
                    bg=self.similarity_color(value),
                    fg=WHITE,
                    width=8,
                    font=("Segoe UI", 8, "bold"),
                )
                cell.grid(row=row, column=col + 1, padx=1, pady=1)

        average_similarity = average_pairwise_similarity(matrix)
        avg_label = tk.Label(
            section,
            text=f"Average pairwise similarity: {average_similarity:.2f}",
            bg=BACKGROUND,
            fg=TEXT,
            font=("Segoe UI", 10),
        )
        avg_label.pack(anchor="w", pady=(8, 0))

        individual = self.individual_average_similarities(matrix, valid_results)
        for result, value in individual:
            if value < 0.60:
                warning = tk.Label(
                    section,
                    text=(
                        f"Warning: {shorten_filename(result['path'])} may be a different person or angle. "
                        "Consider deselecting it."
                    ),
                    bg=BACKGROUND,
                    fg=WARNING,
                    font=("Segoe UI", 10),
                    wraplength=820,
                    justify=tk.LEFT,
                )
                warning.pack(anchor="w", pady=(4, 0))

    def similarity_color(self, value: float) -> str:
        if value < 0.55:
            return DANGER
        if value <= 0.70:
            return WARNING
        return OK

    def individual_average_similarities(
        self,
        matrix: np.ndarray,
        valid_results: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], float]]:
        values: list[tuple[dict[str, Any], float]] = []
        for index, result in enumerate(valid_results):
            if matrix.shape[0] <= 1:
                values.append((result, 1.0))
                continue
            other_values = np.delete(matrix[index], index)
            values.append((result, float(np.mean(other_values))))
        return values

    def save_identity(self) -> None:
        selected = [
            result
            for result in self.results
            if result.get("checkbox_var") is not None
            and bool(result["checkbox_var"].get())
            and result.get("embedding") is not None
        ]

        if len(selected) < 3:
            messagebox.showerror("Not enough images", "Select at least 3 usable reference images.")
            return

        try:
            embeddings = np.stack([np.asarray(result["embedding"], dtype=np.float32) for result in selected])
            mean_embedding = embeddings.mean(axis=0)
            norm = float(np.linalg.norm(mean_embedding))
            if norm == 0.0:
                raise ValueError("Mean embedding has zero norm.")
            identity = (mean_embedding / norm).astype(np.float32)
            embedding_norm = float(np.linalg.norm(identity))
            assert abs(embedding_norm - 1.0) < 0.001
        except AssertionError:
            logger.exception("Embedding normalisation assertion failed.")
            messagebox.showerror("Save failed", "Embedding normalisation failed. Try different reference images.")
            return
        except Exception as exc:
            logger.exception("Could not prepare identity embedding.")
            messagebox.showerror("Save failed", f"Could not prepare identity embedding: {exc}")
            return

        matrix = cosine_matrix(embeddings)
        individual_values = []
        for index in range(len(selected)):
            if len(selected) <= 1:
                individual_values.append(1.0)
            else:
                individual_values.append(float(np.mean(np.delete(matrix[index], index))))

        best_index = int(np.argmax(individual_values))
        best_reference = selected[best_index]["path"].name
        average_similarity = average_pairwise_similarity(matrix)
        individual_similarities = {
            result["path"].name: float(value)
            for result, value in zip(selected, individual_values)
        }

        metadata = {
            "saved_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "num_images_used": len(selected),
            "image_files": [result["path"].name for result in selected],
            "individual_similarities": individual_similarities,
            "average_similarity": float(average_similarity),
            "best_reference": best_reference,
            "embedding_norm": embedding_norm,
        }

        try:
            IDENTITY_DIR.mkdir(parents=True, exist_ok=True)
            np.save(IDENTITY_PATH, identity)
            METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.exception("Could not save identity files.")
            messagebox.showerror("Save failed", f"Could not save identity files: {exc}")
            return

        logger.info("Saved identity to %s with norm %.6f.", IDENTITY_PATH, embedding_norm)
        messagebox.showinfo(
            "Identity saved",
            (
                f"Identity saved. Best reference image: {best_reference}.\n"
                f"Embedding norm: {embedding_norm:.3f}. You may now run generate.py."
            ),
        )
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    FaceIdentityExtractor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
