"""Room/wall classes, following the CubiCasa5K room label map."""

CLASS_NAMES = [
    "Background",   # 0
    "Outdoor",      # 1
    "Wall",         # 2
    "Kitchen",      # 3
    "Living Room",  # 4
    "Bedroom",      # 5
    "Bath",         # 6
    "Entry",        # 7
    "Railing",      # 8
    "Storage",      # 9
    "Garage",       # 10
    "Other Room",   # 11 ("Undefined" in CubiCasa5K)
]
NUM_CLASSES = len(CLASS_NAMES)
WALL = 2
# Classes that count as enclosed rooms when extracting room polygons.
ROOM_CLASSES = [3, 4, 5, 6, 7, 9, 10, 11]
IGNORE_INDEX = 255

# RGB colours for visualisation.
PALETTE = [
    (255, 255, 255), (200, 230, 200), (40, 40, 40), (255, 179, 102),
    (153, 204, 255), (204, 153, 255), (102, 204, 204), (255, 230, 128),
    (160, 160, 160), (210, 180, 140), (180, 180, 220), (230, 200, 200),
]
