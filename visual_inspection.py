"""
Visual Inspection & Contact Sheet Generator
===========================================
Generates a visual grid / contact sheet of sample currency notes across all
7 denominations to visually verify crop quality, aspect ratios, and color fidelity.

Usage:
    python visual_inspection.py
"""

from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont


def create_contact_sheet(
    metadata_path: str = "dataset_metadata.json",
    dataset_dir: str = "dataset",
    output_image_path: str = "dataset_contact_sheet.png",
    samples_per_class: int = 4,
    thumb_size: tuple = (120, 80)
):
    meta_path = Path(metadata_path)
    base_dir = Path(dataset_dir)
    
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file '{metadata_path}' not found.")
        
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    classes = meta.get("classes", [])
    records = meta.get("images", [])
    
    # Organize records by class
    class_images = {c: [] for c in classes}
    for r in records:
        cname = r["class_name"]
        if cname in class_images:
            class_images[cname].append(r)
            
    n_classes = len(classes)
    thumb_w, thumb_h = thumb_size
    padding = 10
    header_w = 90
    
    # Calculate total canvas dimensions
    grid_w = header_w + (samples_per_class * (thumb_w + padding)) + padding
    grid_h = (n_classes * (thumb_h + padding)) + padding + 40
    
    # Create white canvas
    canvas = Image.new("RGB", (grid_w, grid_h), color=(245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    
    # Title Banner
    draw.rectangle([0, 0, grid_w, 35], fill=(30, 41, 59))
    draw.text((15, 10), "Indian Currency Dataset - Visual Verification Grid (50 Images / Class)", fill=(255, 255, 255))
    
    # Render rows
    for row_idx, cname in enumerate(classes):
        y_pos = 45 + (row_idx * (thumb_h + padding))
        
        # Class Label Banner
        draw.rounded_rectangle(
            [padding, y_pos, padding + header_w - 10, y_pos + thumb_h],
            radius=6,
            fill=(226, 232, 240),
            outline=(203, 213, 225)
        )
        draw.text((padding + 10, y_pos + thumb_h // 2 - 8), cname, fill=(15, 23, 42))
        
        # Place sample thumbnails
        samples = class_images[cname][:samples_per_class]
        for col_idx, s in enumerate(samples):
            x_pos = header_w + padding + (col_idx * (thumb_w + padding))
            img_path = base_dir / s["relative_path"]
            
            if img_path.exists():
                with Image.open(img_path) as img:
                    thumb = img.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
                    canvas.paste(thumb, (x_pos, y_pos))
                    draw.rectangle([x_pos, y_pos, x_pos + thumb_w, y_pos + thumb_h], outline=(148, 163, 184), width=1)
                    
    canvas.save(output_image_path)
    print(f"[OK] Visual contact sheet generated and saved to: {Path(output_image_path).resolve()}")


if __name__ == "__main__":
    create_contact_sheet()
