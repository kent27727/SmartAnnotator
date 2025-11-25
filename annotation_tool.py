"""
Manual Annotation Tool v3
=========================
Roboflow-like smooth manual annotation interface
- Polygon Segmentation (auto-complete when clicking start point)
- Zoom feature
- Dynamic class addition
- Class names on annotations
- Project-based workflow support

Usage: python annotation_tool.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
import shutil
import math

import cv2
import numpy as np

try:
    import gradio as gr
except ImportError:
    print("❌ Gradio not installed: pip install gradio")
    sys.exit(1)

import config
from project_manager import ProjectManager

# ============================================
# CLASS MANAGER
# ============================================

class ClassManager:
    """Class management"""
    
    def __init__(self, classes: Dict[int, str] = None, classes_file: Path = None):
        self.classes_file = classes_file
        self.classes = classes if classes else {0: "dark_circle"}
        
        if classes_file and classes_file.exists():
            self.load_classes()
    
    def load_classes(self):
        """Load classes from file"""
        if self.classes_file and self.classes_file.exists():
            with open(self.classes_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    name = line.strip()
                    if name:
                        self.classes[i] = name
        
        if not self.classes:
            self.classes = {0: "dark_circle"}
    
    def save_classes(self):
        """Save classes to file"""
        if self.classes_file:
            with open(self.classes_file, 'w', encoding='utf-8') as f:
                for i in sorted(self.classes.keys()):
                    f.write(f"{self.classes[i]}\n")
    
    def add_class(self, name: str) -> int:
        """Add new class"""
        name = name.strip().lower().replace(" ", "_")
        
        # Check if already exists
        for id, existing_name in self.classes.items():
            if existing_name == name:
                return id
        
        # New ID
        new_id = max(self.classes.keys()) + 1 if self.classes else 0
        self.classes[new_id] = name
        self.save_classes()
        return new_id
    
    def get_choices(self) -> List[Tuple[str, int]]:
        """Return choices for dropdown"""
        return [(name, id) for id, name in sorted(self.classes.items())]
    
    def get_name(self, class_id: int) -> str:
        """Get name from class ID"""
        return self.classes.get(class_id, f"class_{class_id}")


# ============================================
# COLORS
# ============================================

COLORS = [
    (46, 204, 113),   # Green
    (231, 76, 60),    # Red
    (52, 152, 219),   # Blue
    (241, 196, 15),   # Yellow
    (155, 89, 182),   # Purple
    (26, 188, 156),   # Turquoise
    (230, 126, 34),   # Orange
    (149, 165, 166),  # Gray
]

# ============================================
# GLOBAL STATE
# ============================================

class AnnotationState:
    """Annotation state"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.current_image_path = None
        self.original_image = None
        self.current_image = None
        self.image_list = []
        self.current_index = 0
        self.polygons = []
        self.current_polygon = []
        self.boxes = []
        self.mode = "polygon"
        self.class_id = 0
        self.annotations_dir = None
        self.raw_images_dir = None
        self.saved_count = 0
        self.image_size = (0, 0)
        self.zoom_level = 1.0
        self.close_threshold = 20
        self.project_name = None
        self.project_config = None

state = AnnotationState()
class_manager = None
project_manager = ProjectManager()

# ============================================
# HELPER FUNCTIONS
# ============================================

def distance(p1: Tuple[float, float], p2: Tuple[float, float], img_size: Tuple[int, int]) -> float:
    """Calculate pixel distance between two points"""
    w, h = img_size
    x1, y1 = p1[0] * w, p1[1] * h
    x2, y2 = p2[0] * w, p2[1] * h
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def draw_annotations_on_image(image: np.ndarray, zoom: float = 1.0) -> np.ndarray:
    """Draw annotations on image"""
    global state, class_manager
    
    if image is None:
        return None
    
    h, w = image.shape[:2]
    
    # Apply zoom
    if zoom != 1.0:
        new_w = int(w * zoom)
        new_h = int(h * zoom)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = new_h, new_w
    
    result = image.copy()
    overlay = image.copy()
    
    # Draw saved polygons
    for poly in state.polygons:
        color = COLORS[poly['class_id'] % len(COLORS)]
        points = [(int(x * w), int(y * h)) for x, y in poly['points']]
        points_np = np.array(points, dtype=np.int32)
        
        # Fill
        cv2.fillPoly(overlay, [points_np], color)
        # Border
        cv2.polylines(result, [points_np], True, color, 2)
        
        # Write class name
        if points and class_manager:
            top_point = min(points, key=lambda p: p[1])
            class_name = class_manager.get_name(poly['class_id'])
            
            (text_w, text_h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            box_x = top_point[0] - text_w // 2
            box_y = top_point[1] - text_h - 10
            
            cv2.rectangle(result, 
                         (box_x - 5, box_y - 5), 
                         (box_x + text_w + 5, box_y + text_h + 5), 
                         color, -1)
            cv2.putText(result, class_name, 
                       (box_x, box_y + text_h), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw saved boxes
    for box in state.boxes:
        color = COLORS[box['class_id'] % len(COLORS)]
        x_c, y_c, bw, bh = box['coords']
        x1 = int((x_c - bw/2) * w)
        y1 = int((y_c - bh/2) * h)
        x2 = int((x_c + bw/2) * w)
        y2 = int((y_c + bh/2) * h)
        
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        if class_manager:
            class_name = class_manager.get_name(box['class_id'])
            cv2.putText(result, class_name, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Current polygon being drawn
    if state.current_polygon:
        color = COLORS[state.class_id % len(COLORS)]
        points = [(int(x * w), int(y * h)) for x, y in state.current_polygon]
        
        # First point - large target circle
        if len(points) >= 1:
            cv2.circle(result, points[0], 15, (255, 255, 255), 2)
            cv2.circle(result, points[0], 12, color, 2)
            cv2.circle(result, points[0], 6, color, -1)
        
        # Other points
        for pt in points[1:]:
            cv2.circle(result, pt, 5, color, -1)
            cv2.circle(result, pt, 7, (255, 255, 255), 1)
        
        # Lines
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(result, points[i], points[i+1], color, 2)
        
        # Point count indicator
        if len(points) >= 3:
            cv2.putText(result, "Click start point to complete", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Apply overlay
    alpha = 0.35
    result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
    
    return result


# ============================================
# PROJECT FUNCTIONS
# ============================================

def get_project_list() -> List[str]:
    """Get project list"""
    return project_manager.list_projects()


def load_project_for_annotation(project_name: str) -> Tuple[str, gr.Dropdown]:
    """Load project for annotation"""
    global state, class_manager
    
    if not project_name:
        return "❌ No project selected!", gr.Dropdown(choices=[])
    
    cfg = project_manager.load_project(project_name)
    if not cfg:
        return f"❌ Could not load project: {project_name}", gr.Dropdown(choices=[])
    
    project_path = project_manager.get_project_path()
    
    # Update state
    state.project_name = project_name
    state.project_config = cfg
    state.annotations_dir = project_path / "manual_annotations"
    state.raw_images_dir = project_path / "raw_images"
    
    # Create directories
    if cfg["model"]["task"] != "classification":
        (state.annotations_dir / "images").mkdir(parents=True, exist_ok=True)
        (state.annotations_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    # Update class manager
    classes = {int(k): v for k, v in cfg.get("classes", {0: "dark_circle"}).items()}
    classes_file = project_path / "classes.txt"
    class_manager = ClassManager(classes=classes, classes_file=classes_file)
    class_manager.save_classes()
    
    # Activate config
    config.set_active_project(project_name)
    
    task = cfg["model"]["task"]
    model = cfg["model"]["family"]
    
    return (
        f"✅ Project loaded: {project_name}\n"
        f"   Model: {model.upper()} {task}\n"
        f"   Classes: {list(classes.values())}",
        gr.Dropdown(choices=class_manager.get_choices(), value=0)
    )


# ============================================
# ANNOTATION FUNCTIONS
# ============================================

def load_images_from_project() -> Tuple[str, any, str, str, str]:
    """Load images from project"""
    global state
    
    if not state.raw_images_dir:
        return "❌ Select a project first!", None, "", "", get_stats()
    
    if not state.raw_images_dir.exists():
        return f"❌ Raw images directory not found: {state.raw_images_dir}", None, "", "", get_stats()
    
    # Find images
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    seen_files: Set[str] = set()
    images = []
    
    for f in state.raw_images_dir.iterdir():
        if f.is_file():
            ext_lower = f.suffix.lower()
            if ext_lower in extensions:
                normalized_name = f.name.lower()
                if normalized_name not in seen_files:
                    seen_files.add(normalized_name)
                    images.append(f)
    
    if not images:
        return f"❌ No images found: {state.raw_images_dir}", None, "", "", get_stats()
    
    state.image_list = sorted(images, key=lambda x: x.name.lower())
    state.current_index = 0
    
    status = f"✅ {len(images)} images loaded!"
    
    img, info, path = load_current_image()
    
    return status, img, info, path, get_stats()


def load_images_from_directory(directory: str) -> Tuple[str, any, str, str, str]:
    """Load images from directory"""
    global state
    
    if not directory:
        return "❌ No directory specified!", None, "", "", get_stats()
    
    dir_path = Path(directory)
    if not dir_path.exists():
        return f"❌ Directory not found: {directory}", None, "", "", get_stats()
    
    # Find images
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    seen_files: Set[str] = set()
    images = []
    
    for f in dir_path.iterdir():
        if f.is_file():
            ext_lower = f.suffix.lower()
            if ext_lower in extensions:
                normalized_name = f.name.lower()
                if normalized_name not in seen_files:
                    seen_files.add(normalized_name)
                    images.append(f)
    
    if not images:
        return f"❌ No images found: {directory}", None, "", "", get_stats()
    
    state.image_list = sorted(images, key=lambda x: x.name.lower())
    state.current_index = 0
    state.raw_images_dir = dir_path
    
    # Set annotation directory (default if no project)
    if not state.annotations_dir:
        state.annotations_dir = config.MANUAL_ANNOTATIONS_DIR
        (state.annotations_dir / "images").mkdir(parents=True, exist_ok=True)
        (state.annotations_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    status = f"✅ {len(images)} images loaded!"
    
    img, info, path = load_current_image()
    
    return status, img, info, path, get_stats()


def load_current_image() -> Tuple[any, str, str]:
    """Load current image"""
    global state
    
    if not state.image_list:
        return None, "No image loaded", ""
    
    img_path = state.image_list[state.current_index]
    state.current_image_path = img_path
    
    image = cv2.imread(str(img_path))
    if image is None:
        return None, "Could not read image", ""
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    state.original_image = image.copy()
    state.current_image = image
    state.image_size = (image.shape[1], image.shape[0])
    
    load_existing_annotations(img_path)
    
    display = draw_annotations_on_image(image, state.zoom_level)
    
    info = f"📷 {state.current_index + 1}/{len(state.image_list)}: {img_path.name}"
    
    return display, info, str(img_path)


def load_existing_annotations(img_path: Path):
    """Load existing annotations"""
    global state
    
    state.polygons = []
    state.boxes = []
    state.current_polygon = []
    
    if not state.annotations_dir:
        return
    
    label_path = state.annotations_dir / "labels" / (img_path.stem + ".txt")
    
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                
                if len(coords) == 4:
                    state.boxes.append({
                        'class_id': class_id,
                        'coords': coords
                    })
                else:
                    polygon = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                    state.polygons.append({
                        'class_id': class_id,
                        'points': polygon
                    })


def update_zoom(zoom_value: float) -> any:
    """Update zoom level"""
    global state
    
    state.zoom_level = zoom_value
    
    if state.current_image is not None:
        return draw_annotations_on_image(state.current_image, state.zoom_level)
    return None


def handle_click(image, evt: gr.SelectData, mode: str, class_id: int) -> Tuple[any, str]:
    """Handle click on image"""
    global state
    
    if state.current_image is None:
        return image, "❌ Load an image first!"
    
    state.mode = mode
    state.class_id = int(class_id) if class_id is not None else 0
    
    x, y = evt.index[0], evt.index[1]
    
    orig_w, orig_h = state.image_size
    zoomed_w = int(orig_w * state.zoom_level)
    zoomed_h = int(orig_h * state.zoom_level)
    
    x_norm = x / zoomed_w
    y_norm = y / zoomed_h
    
    x_norm = max(0, min(1, x_norm))
    y_norm = max(0, min(1, y_norm))
    
    new_point = (x_norm, y_norm)
    
    if mode == "polygon":
        if len(state.current_polygon) >= 3:
            first_point = state.current_polygon[0]
            dist = distance(new_point, first_point, state.image_size)
            
            if dist < state.close_threshold:
                state.polygons.append({
                    'class_id': state.class_id,
                    'points': state.current_polygon.copy()
                })
                state.current_polygon = []
                
                display = draw_annotations_on_image(state.current_image, state.zoom_level)
                class_name = class_manager.get_name(state.class_id) if class_manager else f"class_{state.class_id}"
                return display, f"✅ '{class_name}' polygon completed! (Total: {len(state.polygons)})"
        
        state.current_polygon.append(new_point)
        
        display = draw_annotations_on_image(state.current_image, state.zoom_level)
        
        if len(state.current_polygon) < 3:
            status = f"✏️ {len(state.current_polygon)} points (min 3 required)"
        else:
            status = f"✏️ {len(state.current_polygon)} points - 🎯 Click start point to complete"
        
        return display, status
    
    return image, "📦 Box mode - coming soon"


def complete_polygon() -> Tuple[any, str]:
    """Manually complete polygon"""
    global state
    
    if len(state.current_polygon) < 3:
        display = draw_annotations_on_image(state.current_image, state.zoom_level) if state.current_image is not None else None
        return display, "❌ At least 3 points required!"
    
    state.polygons.append({
        'class_id': state.class_id,
        'points': state.current_polygon.copy()
    })
    state.current_polygon = []
    
    display = draw_annotations_on_image(state.current_image, state.zoom_level)
    class_name = class_manager.get_name(state.class_id) if class_manager else f"class_{state.class_id}"
    
    return display, f"✅ '{class_name}' added! (Total: {len(state.polygons)})"


def undo_last() -> Tuple[any, str]:
    """Undo last action"""
    global state
    
    if state.current_polygon:
        state.current_polygon.pop()
        status = f"↩️ Last point removed ({len(state.current_polygon)} remaining)"
    elif state.polygons:
        removed = state.polygons.pop()
        class_name = class_manager.get_name(removed['class_id']) if class_manager else f"class_{removed['class_id']}"
        status = f"↩️ '{class_name}' removed ({len(state.polygons)} remaining)"
    elif state.boxes:
        state.boxes.pop()
        status = f"↩️ Last box removed ({len(state.boxes)} remaining)"
    else:
        display = draw_annotations_on_image(state.current_image, state.zoom_level) if state.current_image is not None else None
        return display, "❌ Nothing to undo!"
    
    display = draw_annotations_on_image(state.current_image, state.zoom_level) if state.current_image is not None else None
    return display, status


def clear_current() -> Tuple[any, str]:
    """Clear current drawing"""
    global state
    
    state.current_polygon = []
    display = draw_annotations_on_image(state.current_image, state.zoom_level) if state.current_image is not None else None
    
    return display, "🗑️ Drawing cleared"


def clear_all() -> Tuple[any, str]:
    """Clear all annotations"""
    global state
    
    state.polygons = []
    state.boxes = []
    state.current_polygon = []
    
    display = draw_annotations_on_image(state.current_image, state.zoom_level) if state.current_image is not None else None
    
    return display, "🗑️ All annotations cleared"


def save_annotation() -> str:
    """Save annotation"""
    global state
    
    if state.current_image_path is None:
        return "❌ No image loaded!"
    
    if not state.polygons and not state.boxes:
        return "❌ No annotations to save!"
    
    if not state.annotations_dir:
        return "❌ Annotation directory not set!"
    
    # Copy image
    img_dest = state.annotations_dir / "images" / state.current_image_path.name
    if not img_dest.exists():
        shutil.copy(state.current_image_path, img_dest)
    
    # Label file
    label_path = state.annotations_dir / "labels" / (state.current_image_path.stem + ".txt")
    
    with open(label_path, 'w') as f:
        for poly in state.polygons:
            class_id = poly['class_id']
            coords = []
            for x, y in poly['points']:
                coords.extend([f"{x:.6f}", f"{y:.6f}"])
            f.write(f"{class_id} " + " ".join(coords) + "\n")
        
        for box in state.boxes:
            class_id = box['class_id']
            x_c, y_c, w, h = box['coords']
            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
    
    state.saved_count += 1
    
    return f"✅ Saved! ({state.saved_count} total)"


def save_and_next() -> Tuple[any, str, str, str, str]:
    """Save and go to next"""
    global state
    
    save_msg = save_annotation()
    
    if state.image_list:
        state.current_index = (state.current_index + 1) % len(state.image_list)
    
    img, info, path = load_current_image()
    
    return img, save_msg + " → Next", info, path, get_stats()


def next_image() -> Tuple[any, str, str]:
    """Next image"""
    global state
    
    if not state.image_list:
        return None, "No image loaded", ""
    
    state.current_index = (state.current_index + 1) % len(state.image_list)
    return load_current_image()


def prev_image() -> Tuple[any, str, str]:
    """Previous image"""
    global state
    
    if not state.image_list:
        return None, "No image loaded", ""
    
    state.current_index = (state.current_index - 1) % len(state.image_list)
    return load_current_image()


def goto_image(index: int) -> Tuple[any, str, str]:
    """Go to specific image"""
    global state
    
    if not state.image_list:
        return None, "No image loaded", ""
    
    index = int(index) - 1
    if 0 <= index < len(state.image_list):
        state.current_index = index
    
    return load_current_image()


def add_new_class(class_name: str) -> Tuple[gr.Dropdown, str]:
    """Add new class"""
    global class_manager
    
    if class_manager is None:
        return gr.Dropdown(choices=[("dark_circle", 0)]), "❌ Select a project first!"
    
    if not class_name or not class_name.strip():
        return gr.Dropdown(choices=class_manager.get_choices()), "❌ Class name cannot be empty!"
    
    new_id = class_manager.add_class(class_name)
    new_name = class_manager.get_name(new_id)
    
    return gr.Dropdown(choices=class_manager.get_choices(), value=new_id), f"✅ '{new_name}' class added (ID: {new_id})"


def get_stats() -> str:
    """Statistics"""
    global state, class_manager
    
    if not state.annotations_dir:
        return "📊 No project selected"
    
    labels_dir = state.annotations_dir / "labels"
    saved = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0
    
    total = len(state.image_list)
    remaining = max(0, total - saved)
    progress = (saved / total * 100) if total > 0 else 0
    
    classes_str = ""
    if class_manager:
        classes_str = ", ".join([f"{name}" for id, name in sorted(class_manager.classes.items())])
    
    project_info = f"**Project:** {state.project_name}" if state.project_name else "**Project:** Not selected"
    
    return f"""📊 **Statistics**
━━━━━━━━━━━━━━━━━━
{project_info}

📁 Total: **{total}**
✅ Labeled: **{saved}**
⏳ Remaining: **{remaining}**
📈 Progress: **{progress:.1f}%**

**Current Image:**
🔷 Polygons: {len(state.polygons)}
📦 Boxes: {len(state.boxes)}
✏️ Drawing: {len(state.current_polygon)} points

**Classes:** {classes_str}"""


# ============================================
# GRADIO GUI
# ============================================

def create_annotation_gui():
    """Annotation GUI"""
    global class_manager
    
    # Default class manager
    if class_manager is None:
        class_manager = ClassManager()
    
    with gr.Blocks(title="Manual Annotation Tool v3", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🎯 Manual Annotation Tool v3
        **Click to draw polygon, click on start point to complete**
        """)
        
        with gr.Row():
            # Left Panel
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Project Selection")
                project_dropdown = gr.Dropdown(
                    choices=get_project_list(),
                    label="Project",
                    value=None
                )
                load_project_btn = gr.Button("📂 Load Project", variant="primary", size="sm")
                project_status = gr.Textbox(label="Project Status", interactive=False, lines=3)
                
                gr.Markdown("---")
                gr.Markdown("### 📂 Image Source")
                dir_input = gr.Textbox(
                    label="Custom Directory (optional)",
                    placeholder="Project raw_images will be used",
                    lines=1
                )
                with gr.Row():
                    load_project_images_btn = gr.Button("📷 Project Images", size="sm")
                    load_custom_btn = gr.Button("📂 Custom Dir", size="sm")
                load_status = gr.Textbox(label="Status", interactive=False, lines=1)
                
                gr.Markdown("---")
                gr.Markdown("### 🔍 Zoom")
                zoom_slider = gr.Slider(
                    minimum=0.5,
                    maximum=3.0,
                    value=1.0,
                    step=0.1,
                    label="Zoom"
                )
                
                gr.Markdown("---")
                gr.Markdown("### 🎨 Class Selection")
                
                class_dropdown = gr.Dropdown(
                    choices=class_manager.get_choices() if class_manager else [("dark_circle", 0)],
                    value=0,
                    label="Class"
                )
                
                with gr.Row():
                    new_class_input = gr.Textbox(
                        label="New Class",
                        placeholder="e.g., wrinkle",
                        scale=3
                    )
                    add_class_btn = gr.Button("➕", size="sm", scale=1)
                
                mode_radio = gr.Radio(
                    choices=["polygon", "box"],
                    value="polygon",
                    label="Drawing Mode"
                )
                
                gr.Markdown("---")
                gr.Markdown("### 🛠️ Tools")
                
                complete_btn = gr.Button("✅ Complete Polygon", size="sm")
                
                with gr.Row():
                    undo_btn = gr.Button("↩️ Undo", size="sm")
                    clear_btn = gr.Button("🗑️ Clear", size="sm")
                
                clear_all_btn = gr.Button("🗑️ Clear All", variant="stop", size="sm")
                
                gr.Markdown("---")
                gr.Markdown("### 💾 Save")
                
                with gr.Row():
                    save_btn = gr.Button("💾 Save", variant="primary")
                    save_next_btn = gr.Button("💾 Save →", variant="secondary")
                
                skip_btn = gr.Button("⏭️ Skip (Don't Save)", size="sm")
                
                gr.Markdown("---")
                stats_md = gr.Markdown(get_stats())
            
            # Right Panel
            with gr.Column(scale=4):
                with gr.Row():
                    prev_btn = gr.Button("← Previous", size="sm")
                    image_info = gr.Textbox(
                        label="", 
                        interactive=False, 
                        scale=3,
                        show_label=False
                    )
                    next_btn = gr.Button("Next →", size="sm")
                
                with gr.Row():
                    goto_num = gr.Number(label="Go to:", value=1, minimum=1, scale=1, precision=0)
                    goto_btn = gr.Button("→", size="sm", scale=0)
                    image_path = gr.Textbox(label="File", interactive=False, scale=4)
                
                image_display = gr.Image(
                    label="🖼️ Image",
                    interactive=True,
                    height=600
                )
                
                status_bar = gr.Textbox(
                    label="Status", 
                    interactive=False,
                    max_lines=1
                )
        
        gr.Markdown("""
        ---
        **💡 Tips:** 🎯 Click start point (white ring) to complete | 
        🔍 Use zoom slider to zoom in | ➕ You can add new classes
        """)
        
        # Events
        load_project_btn.click(
            fn=load_project_for_annotation,
            inputs=[project_dropdown],
            outputs=[project_status, class_dropdown]
        )
        
        load_project_images_btn.click(
            fn=load_images_from_project,
            outputs=[load_status, image_display, image_info, image_path, stats_md]
        )
        
        load_custom_btn.click(
            fn=load_images_from_directory,
            inputs=[dir_input],
            outputs=[load_status, image_display, image_info, image_path, stats_md]
        )
        
        zoom_slider.change(
            fn=update_zoom,
            inputs=[zoom_slider],
            outputs=[image_display]
        )
        
        image_display.select(
            fn=handle_click,
            inputs=[image_display, mode_radio, class_dropdown],
            outputs=[image_display, status_bar]
        )
        
        complete_btn.click(
            fn=complete_polygon,
            outputs=[image_display, status_bar]
        )
        
        undo_btn.click(
            fn=undo_last,
            outputs=[image_display, status_bar]
        )
        
        clear_btn.click(
            fn=clear_current,
            outputs=[image_display, status_bar]
        )
        
        clear_all_btn.click(
            fn=clear_all,
            outputs=[image_display, status_bar]
        )
        
        save_btn.click(
            fn=save_annotation,
            outputs=[status_bar]
        ).then(
            fn=get_stats,
            outputs=[stats_md]
        )
        
        save_next_btn.click(
            fn=save_and_next,
            outputs=[image_display, status_bar, image_info, image_path, stats_md]
        )
        
        skip_btn.click(
            fn=next_image,
            outputs=[image_display, image_info, image_path]
        )
        
        next_btn.click(
            fn=next_image,
            outputs=[image_display, image_info, image_path]
        )
        
        prev_btn.click(
            fn=prev_image,
            outputs=[image_display, image_info, image_path]
        )
        
        goto_btn.click(
            fn=goto_image,
            inputs=[goto_num],
            outputs=[image_display, image_info, image_path]
        )
        
        add_class_btn.click(
            fn=add_new_class,
            inputs=[new_class_input],
            outputs=[class_dropdown, status_bar]
        )
    
    return demo


def main():
    """Main function"""
    global class_manager
    
    print("\n" + "="*60)
    print("🎯 MANUAL ANNOTATION TOOL v3")
    print("="*60)
    
    # Default class manager
    class_manager = ClassManager()
    
    # List projects
    projects = project_manager.list_projects()
    if projects:
        print(f"\n📂 Available projects: {', '.join(projects)}")
    else:
        print("\n💡 No projects yet. You can select project from interface.")
    
    demo = create_annotation_gui()
    
    print("\n🌐 Starting: http://localhost:7861")
    print("\n💡 Features:")
    print("   - Project-based workflow")
    print("   - Zoom slider")
    print("   - Click start point to complete polygon")
    print("   - Add new classes")
    print("   - Class name display on annotations")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
