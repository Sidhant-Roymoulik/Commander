from BattleshipBoardV1 import BattleshipBoardV1

if __name__ == "__main__":
    board = BattleshipBoardV1()

    print("Ships (internal view):")
    board.display_ships()

    # attack every other cell on the board
    for r in range(board.rows):
        for c in range(board.cols):
            if (r + c) % 2 == 0:
                try:
                    board.attack(r, c)
                except ValueError:
                    pass  # ignore already attacked errors

    try:
        board.attack(100, 100)  # out of bounds
    except ValueError as e:
        print(f"Error: {e}")

    try:
        board.attack(2, 4)  # already attacked
    except ValueError as e:
        print(f"Error: {e}")

    print("\nCombined board (hits = X, misses = O):")
    board.display_combined()
