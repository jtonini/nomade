#!/usr/bin/env python3
"""Combine dashboard screenshots into a labeled panel."""

from PIL import Image, ImageDraw, ImageFont
import os

# Image files (update paths as needed)
images = [
    ('compute.png', 'A'),      # Top-left
    ('highmem.png', 'B'),      # Top-right
    ('gpu.png', 'C'),          # Bottom-left
    ('network.png', 'D'),      # Bottom-right
]

# Load images
imgs = []
for fname, label in images:
    if os.path.exists(fname):
        imgs.append((Image.open(fname), label))
    else:
        print(f"Missing: {fname}")

if len(imgs) != 4:
    print("Need 4 images")
    exit(1)

# Get dimensions
w, h = imgs[0][0].size

# Create 2x2 panel with white gap
gap = 20
panel_w = w * 2 + gap
panel_h = h * 2 + gap

# White background
panel = Image.new('RGB', (panel_w, panel_h), color='white')

# Place images
positions = [(0, 0), (w + gap, 0), (0, h + gap), (w + gap, h + gap)]

for (img, label), (x, y) in zip(imgs, positions):
    # Resize if needed
    if img.size != (w, h):
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    panel.paste(img, (x, y))
    
    # Add label
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 52)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
        except:
            font = ImageFont.load_default()
    
    # Position label in top-left corner of each image
    label_x = x + 14
    label_y = y + 10
    
    # Draw white box with thin dark border
    bbox = draw.textbbox((label_x, label_y), label, font=font)
    padding = 8
    box_coords = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]
    draw.rectangle(box_coords, fill='white', outline='#333333', width=2)
    draw.text((label_x, label_y), label, font=font, fill='black')

panel.save('dashboard_panel.png', dpi=(150, 150))
print(f"Created dashboard_panel.png ({panel_w}x{panel_h})")
