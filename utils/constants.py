# Board symbols
WATER = "~"
SHIP = "S"
HIT = "X"
MISS = "O"

# Default board size
DEFAULT_ROWS = 10
DEFAULT_COLS = 10

# Default ship sizes (standard Battleship)
DEFAULT_SHIP_SIZES = [2, 3, 3, 4, 5]

# Ship board empty value (use integers for ship ids)
SHIP_EMPTY = 0

# ML Representation constants
ML_DTYPE = "float32"
ML_OCCUPIED = 1.0  # Normalized value for occupied/hit/miss
ML_EMPTY = 0.0  # Normalized value for empty
