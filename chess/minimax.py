import time

import web_parser
from chess_logic import GameState, Piece
from collections import defaultdict
import random, copy

prev_data = None
awaiting_move = False
data = bytearray(b'\x01\x01\x01\x01\x00\x00\x01')
memo = {}
nodes_explored = 0

piece_values = {1: ("Pawn", 100, 0), 2: ("Knight", 325, 5), 3: ("Bishop", 330, 4),
          4: ("Rook", 500, 3), 5: ("Queen", 950, 2), 6: ("King", 0, 0)}


def reconstructState(vector: bytes) -> GameState:
    state = GameState()

    # Clear the default empty set if needed, though init makes it empty
    state.board = set()
    state.lookup = defaultdict(lambda: None)

    i = 0
    while vector[i] != 0:
        p = Piece((vector[i + 2] > 0), abs(vector[i + 2]), vector[i], vector[i + 1])
        state.board.add(p)  # Changed from append() to add()
        state.lookup[(vector[i], vector[i + 1])] = p  # Directly use p, no indexing needed
        i += 3

    state.metadata = bytearray(vector[i + 1:])

    return state


def moveScore(state, action):
    src, dst = action
    piece = state.pieceAt(((src % 8) + 1, (src // 8) + 1))
    captured = state.pieceAt(((dst % 8) + 1, (dst // 8) + 1))

    if captured:
        # Higher values for capturing big pieces with small pieces
        return (piece_values[captured.data[1]][1] * 10) - piece_values[piece.data[1]][1]
    return 0

def negamax(state, depth, alpha, beta, color):
    """
    color = +1 if engine plays side-to-move
            -1 otherwise
    """
    global nodes_explored

    hash = state.hash
    nodes_explored += 1

    # ----------------------------
    # Transposition table lookup
    # ----------------------------
    entry = memo.get(hash)
    if entry and entry[0] >= depth:
        return entry[1], None

    moves = state.getAllLegalActions(state.sideToMove())

    # ----------------------------
    # Terminal / leaf
    # ----------------------------
    if depth == 0 or not moves:
        score = color * state.getScore()
        memo[hash] = (depth, score)
        return score, None

    # ----------------------------
    # Move ordering (CRITICAL)
    # ----------------------------
    moves.sort(key=lambda m: moveScore(state, m), reverse=True)

    best_move = None
    best_value = -99999999

    for move in moves:
        undo = state.makeMove(move)

        value, _ = negamax(state, depth - 1, -beta, -alpha, -color)
        value = -value

        state.undoMove(undo)

        if value > best_value:
            best_value = value
            best_move = move

        alpha = max(alpha, value)
        if alpha >= beta:
            break

    # ----------------------------
    # Store in TT
    # ----------------------------
    memo[hash] = (depth, best_value)

    return best_value, best_move


if __name__ == '__main__':
    web_parser.open_site()
    nodes_explored = 0

    playerIsWhite = web_parser.determine_side()

    def decodeAction(state, action):

        src_x = (action[0] % 8) + 1
        src_y = (action[0] // 8) + 1

        dst_x = (action[1] % 8) + 1
        dst_y = (action[1] // 8) + 1

        piece = state.pieceAt((src_x, src_y))

        return f"{piece.__str__()} to {chr(dst_x+96)}{dst_y}"



    while True:
        time.sleep(1.5)
        pieces = web_parser.get_board_data()
        state = GameState(pieces)
        whiteMovedLast = web_parser.determine_last_moved() == 'w'
        state.metadata[6] = 1 if not whiteMovedLast else 0

        if state != prev_data and whiteMovedLast != playerIsWhite:
            prev_data = state


            nodes_explored = 0
            print("Calculating move...")
            result, action = negamax(state, 6, -9999999, 9999999, playerIsWhite)
            awaiting_move = True
            if action:
                print("Nodes explored: " + str(nodes_explored))
                print(decodeAction(state, action))
            else:
                print("No valid moves")




