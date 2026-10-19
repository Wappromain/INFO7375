"""
make_grid_template.py
Writes grid_template.png: a printable 10 x 10 grid with row labels in the left
margin. Write ten versions of each digit in its row, then photograph the sheet
and crop the photo to the OUTER EDGE OF THE GRID (leave out the labels).

    python make_grid_template.py
"""
from PIL import Image, ImageDraw, ImageFont

CELL = 220                     # pixels per cell; about 1.8 cm when printed on Letter/A4
LEFT, TOP = 160, 120
W, H = LEFT + 10 * CELL + 60, TOP + 10 * CELL + 60

img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
def get_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)     # Pillow 10.1 or newer
    except TypeError:
        return ImageFont.load_default()


font = get_font(70)
d.text((LEFT, 25), "Write each digit 10 times in its row. Crop photo to the grid.",
       fill=(120, 120, 120), font=get_font(46))
for i in range(11):
    d.line([(LEFT + i * CELL, TOP), (LEFT + i * CELL, TOP + 10 * CELL)], fill=(170, 170, 170), width=3)
    d.line([(LEFT, TOP + i * CELL), (LEFT + 10 * CELL, TOP + i * CELL)], fill=(170, 170, 170), width=3)
for r in range(10):
    d.text((50, TOP + r * CELL + CELL // 2 - 40), str(r), fill=(150, 150, 150), font=font)

img.save("grid_template.png")
print("Wrote grid_template.png. Print it, fill it in, photograph it.")
