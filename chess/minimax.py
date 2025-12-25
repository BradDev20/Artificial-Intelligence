import time
import web_parser
from chess_logic import GameState

prev_data = None
awaiting_move = False
data = bytearray(b'\x01\x01\x01\x01\x00\x00\x01')
killers = [[None, None] for _ in range(32)]

EXACT = 0
LOWER = 1
UPPER = 2

memo = {}
nodes_explored = 0

piece_values = {1: ("Pawn", 100, 0), 2: ("Knight", 325, 5), 3: ("Bishop", 330, 4),
          4: ("Rook", 500, 3), 5: ("Queen", 950, 2), 6: ("King", 0, 0)}

def see(state, move):
    src, dst = move
    ax = (src % 8) + 1
    ay = (src // 8) + 1
    dx = (dst % 8) + 1
    dy = (dst // 8) + 1

    attacker = state.pieceAt((ax, ay))
    captured = state.pieceAt((dx, dy))

    if not captured:
        return -99999999

    return piece_values[captured[1]][1] - piece_values[attacker[1]][1]

def _moveScore(state, action):
    src, dst = action
    piece = state.pieceAt(((src % 8) + 1, (src // 8) + 1))
    captured = state.pieceAt(((dst % 8) + 1, (dst // 8) + 1))

    if captured and piece:
        # Higher values for capturing big pieces with small pieces
        return (piece_values[captured[1]][1] * 10) - piece_values[piece[1]][1]
    return 0

def _orderMoves(state, moves, depth, memo_move):
    ordered = []

    # 1. Memo move
    if memo_move and memo_move in moves:
        ordered.append(memo_move)
        moves.remove(memo_move)

    # 2. Killer moves
    for km in killers[depth]:
        if km and km in moves:
            ordered.append(km)
            moves.remove(km)

    # 3. Captures
    captures = []
    quiet = []
    for m in moves:
        src, dst = m
        if state.pieceAt(((dst % 8) + 1, (dst // 8) + 1)):
            captures.append(m)
        else:
            quiet.append(m)

    # sort captures by SEE order
    captures.sort(key=lambda m: see(state, m), reverse=True)

    ordered.extend(captures)
    ordered.extend(quiet)
    return ordered


def _quiescence(state, alpha, beta, color):
    """
    Performs quiescence search on the GameState.
    :param state: the GameState to be analyzed
    :param alpha: alpha value(from negamax function)
    :param beta: beta value(from negamax function)
    :param color: whether White is being analyzed
    :return:
    """
    stand_pat = color * state.getScore()

    if stand_pat >= beta:
        return beta

    alpha = max(alpha, stand_pat)

    moves = state.getAllLegalActions(state.sideToMove())
    for move in moves:

        # Don't look at 'noisy' moves(piece is captured)
        src, dst = move
        if not state.pieceAt(((dst % 8) + 1, (dst // 8) + 1)):
            continue

        undo = state.makeMove(move)

        if state.kingInCheck(not state.sideToMove()):
            state.undoMove(undo)
            continue

        score = -_quiescence(state, -beta, -alpha, -color)
        state.undoMove(undo)

        if score >= beta:
            return beta
        alpha = max(alpha, score)

    return alpha

def negamax(state, depth, alpha, beta, color):
    global nodes_explored
    nodes_explored += 1

    alpha_orig = alpha
    h = state.hash

    # Memo lookup
    entry = memo.get(h)
    if entry:
        d, val, flag, best = entry
        if d >= depth:

            # Normal
            if flag == EXACT:
                return val, best

            # Alpha-pruned node
            elif flag == LOWER:
                alpha = max(alpha, val)

            # Beta-pruned node
            elif flag == UPPER:
                beta = min(beta, val)

            if alpha >= beta:
                return val, best

    # Leaf
    if depth == 0:
        return _quiescence(state, alpha, beta, color), None

    # Null pruning
    # If state >= beta after 'skipping' a turn, then terminate the branch
    # only done if shallow enough, the king is safe, and not in endgame
    if depth >= 3 and not state.kingInCheck(state.sideToMove()) and not state.isEndgame():
        undo = state.makeNullMove()
        score, _ = negamax(state, depth - 1 - 2, -beta, -beta + 1, -color)
        score = -score
        state.undoNullMove(undo)

        if score >= beta:
            return beta, None

    moves = state.getAllLegalActions(state.sideToMove())
    if not moves:
        return color * state.getScore(), None

    memo_move = entry[3] if entry else None
    moves = _orderMoves(state, moves, depth, memo_move)

    best_value = -10**9
    best_move = None

    for move in moves:
        undo = state.makeMove(move)

        if state.kingInCheck(not state.sideToMove()):
            state.undoMove(undo)
            continue

        value, _ = negamax(state, depth - 1, -beta, -alpha, -color)
        value = -value

        state.undoMove(undo)

        if value > best_value:
            best_value = value
            best_move = move

        alpha = max(alpha, value)

        if alpha >= beta:
            # 'killer' moves are moves that do not capture material
            # However, they do cause a beta prune
            # This means they are effective and should be prioritized
            dst = ((move[1] % 8) + 1, (move[1] // 8) + 1)
            if not state.pieceAt(dst):
                if move != killers[depth][0]:
                    killers[depth][1] = killers[depth][0]
                    killers[depth][0] = move
            break

    # Memoize state result and flag
    # Prevents redundant state calculation
    if best_value <= alpha_orig:
        flag = UPPER
    elif best_value >= beta:
        flag = LOWER
    else:
        flag = EXACT

    memo[h] = (depth, best_value, flag, best_move)

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

        if piece[0]:
            color = "White"
        else:
            color = "Black"
        result = f"{color} {piece_values[piece[1]][0]} at {chr(piece[2]+96)}{piece[3]}"

        return f"{result} to {chr(dst_x+96)}{dst_y}"



    while True:
        time.sleep(1.5)
        pieces = web_parser.get_board_data()
        state = GameState(pieces)
        whiteMovedLast = web_parser.determine_last_moved() == 'w'
        state.metadata[6] = 1 if not whiteMovedLast else 0

        if state != prev_data and whiteMovedLast != playerIsWhite:
            prev_data = state


            nodes_explored = 0
            start = time.time()
            print("Calculating move...")
            result, action = negamax(state, 6, -9999999, 9999999, playerIsWhite)
            awaiting_move = True
            if action:
                print("Nodes explored: " + str(nodes_explored))
                end = time.time() - start
                #end = (end * 1000) // 1000
                print("Execution time: " + str(end) + " seconds")
                print(decodeAction(state, action))
                print('\n')
            else:
                print("No valid moves")




