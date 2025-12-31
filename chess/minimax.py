import time
import web_parser, cProfile, pstats
from chess_logic import GameState


prev_data = None
data = bytearray(b'\x01\x01\x01\x01\x00\x00\x01')
killers = [[None, None] for _ in range(32)]

EXACT = 0
LOWER = 1
UPPER = 2

memo = {}
nodes_explored = 0

piece_values = (0, 100, 325, 330, 500, 950, 16383)
names = ["", "Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]

def mvv_lva(state, move) -> int:
    """
    Most Valuable Victim - Least Valuable Attacker heuristic.
    :return: higher score for good captures (big piece captured by small piece)
    """
    src, dst = move
    attacker = state.pieceAt(((src % 8) + 1, (src // 8) + 1))
    victim = state.pieceAt(((dst % 8) + 1, (dst // 8) + 1))
    if attacker is None or victim is None:
        return 0
    return 10 * piece_values[victim[1]] - piece_values[attacker[1]]

def _orderMoves(state, moves, depth, memo_move):
    """
    Orders moves for better alpha-beta efficiency.

    Ordering priority (highest first):
    1. Hash table best move (if available)
    2. Killer moves at current depth
    3. Good captures (sorted by MVV-LVA)
    4. Quiet moves (in original order)

    :param state: current GameState
    :param moves: list of raw legal moves (src,dst 0-63)
    :param depth: current search depth (for killers)
    :param memo_move: best move from memo (if any)

    :return: list of moves in recommended search order
    """
    ordered = []

    # 1. Memo move
    if memo_move and memo_move in moves and state.isMovePseudoLegal(memo_move):
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
        if state.pieceAt(((dst % 8) + 1, (dst // 8) + 1)) is not None:
            captures.append(m)
        else:
            quiet.append(m)

    # sort captures by SEE order
    captures.sort(key=lambda m: mvv_lva(state, m), reverse=True)

    ordered.extend(captures)
    ordered.extend(quiet)
    return ordered


def _quiescence(state, alpha, beta, color) -> int:
    """
    Performs quiescence search on the GameState.
    :param state: the GameState to be analyzed
    :param alpha: alpha value(from negamax function)
    :param beta: beta value(from negamax function)
    :param color: whether White is being analyzed
    :return:
    """
    stand_pat = state.getScore()

    if stand_pat >= beta:
        return beta

    alpha = max(alpha, stand_pat)

    moves = state.getAllLegalActions(state.sideToMove())

    for move in moves:

        # Only look at legal 'noisy' moves(piece is captured)
        src, dst = move
        if state.pieceAt(((dst % 8) + 1, (dst // 8) + 1)) is None or state.pieceAt(((src % 8) + 1, (src // 8) + 1)) is None:
            continue

        if state.see(src, dst) < 0:
            continue

        undo = state.makeMove(move)
        score = -_quiescence(state, -beta, -alpha, -color)
        state.undoMove(undo)

        if score >= beta:
            return beta
        alpha = max(alpha, score)

    return alpha


def negamax(state, depth, alpha, beta, color):
    """
    Main search function - negamax with alpha-beta pruning,
    transposition table, null move pruning, move ordering and killer heuristic.

    :param state:      current GameState (will be modified & restored)
    :param depth:      remaining plies to search
    :param alpha:      best score maximizer can guarantee
    :param beta:       best score minimizer can guarantee
    :param color:      +1 for white-to-move perspective, -1 for black

    :return: (score, best_move) tuple score is from perspective of the side who was to move. best_move is the best move found (0-63 src,dst tuple) or None
    """
    global nodes_explored
    nodes_explored += 1

    # 1. INIT
    alpha_orig = alpha
    beta_orig = beta
    h = state.hash

    # 2. TRANSPOSITION TABLE LOOKUP
    entry = memo.get(h)
    if entry:
        d, val, flag, best = entry
        if d >= depth:
            if flag == EXACT:
                return val, best
            elif flag == LOWER:
                alpha = max(alpha, val)
            elif flag == UPPER:
                beta = min(beta, val)
            if alpha >= beta:
                return val, best

    # 3. BASE CASE: CHECKMATE / STALEMATE DETECTION
    # We must generate moves to know if the game is over.
    moves = state.getAllLegalActions(state.sideToMove())
    #print(moves)

    if not moves:
        if state.kingInCheck(state.sideToMove()):
            # Checkmate
            return -9999999 + depth, None
        else:
            # Stalemate
            return 0, None

    # 4. DEPTH LIMIT / QUIESCENCE
    if depth == 0:
        return _quiescence(state, alpha, beta, color), None


    # 5. NULL MOVE PRUNING (Optional: Disable if debugging)
    if depth >= 3 and not state.kingInCheck(state.sideToMove()) and not state.isEndgame():
        undo = state.makeNullMove()
        # Pass -color and appropriate alpha/beta
        score, _ = negamax(state, depth - 1 - 2, -beta, -beta + 1, -color)
        score = -score
        state.undoNullMove(undo)
        if score >= beta:
            memo[h] = (depth, beta, LOWER, None)
            return beta, None



    # 6. MOVE ORDERING
    memo_move = entry[3] if entry else None
    moves = _orderMoves(state, moves, depth, memo_move)

    best_value = -float('inf')
    best_move = None
    # 7. RECURSION
    for move in moves:
        undo = state.makeMove(move)

        # Verify legality (king safety)
        if state.kingInCheck(not state.sideToMove()):
            state.undoMove(undo)
            continue

        # Recursive call: decrease depth, flip alpha/beta, flip color
        value, _ = negamax(state, depth - 1, -beta, -alpha, -color)
        value = -value

        state.undoMove(undo)

        if value > best_value:
            best_value = value
            best_move = move

        alpha = max(alpha, value)
        if alpha >= beta:
            # Killer Heuristic Update
            dst = ((move[1] % 8) + 1, (move[1] // 8) + 1)
            # Only store quiet moves as killers
            if state.pieceAt(dst) is None:
                if move != killers[depth][0]:
                    killers[depth][1] = killers[depth][0]
                    killers[depth][0] = move
            break

    # 8. STORE IN TABLE
    # If no legal moves were found in the loop (but moves list wasn't empty initially),
    # it means all moves were self-checks. Treat as Checkmate/Stalemate.
    if best_value == -float('inf'):
        if state.kingInCheck(state.sideToMove()):
            return -9999999 + depth, None
        else:
            return 0, None

    if best_value <= alpha_orig:
        flag = UPPER
    elif best_value >= beta_orig:
        flag = LOWER
    else:
        flag = EXACT

    old = memo.get(h)
    if old is None or depth >= old[0]:
        memo[h] = (depth, best_value, flag, best_move)

    return best_value, best_move


if __name__ == '__main__':
    web_parser.open_site()
    nodes_explored = 0

    playerIsWhite = web_parser.determine_side()
    color = 1 if playerIsWhite else -1
    pr = cProfile.Profile()

    def decodeAction(state, action) -> str:
        """
        Converts an action for a given into something readable.
        """

        src_x = (action[0] % 8) + 1
        src_y = (action[0] // 8) + 1

        dst_x = (action[1] % 8) + 1
        dst_y = (action[1] // 8) + 1

        piece = state.pieceAt((src_x, src_y))

        if piece[0]:
            color = "White"
        else:
            color = "Black"
        result = f"{color} {names[piece[1]]} at {chr(piece[2]+96)}{piece[3]}"

        return f"{result} to {chr(dst_x+96)}{dst_y}"



    while True:
        time.sleep(1.5)
        pieces = web_parser.get_board_data()
        #print(pieces)
        state = GameState(pieces)
        whiteMovedLast = web_parser.determine_last_moved() == 'w'
        state.metadata[6] = 1 if not whiteMovedLast else 0

        if state != prev_data and whiteMovedLast != playerIsWhite:
            prev_data = state


            nodes_explored = 0
            start = time.time()
            pr.enable()
            print("Calculating move...")
            result, action = negamax(state, 4, -9999999, 9999999, color)
            awaiting_move = True
            pr.disable()
            if action:
                print("Nodes explored: " + str(nodes_explored))
                end = time.time() - start
                #end = (end * 1000) // 1000
                print("Execution time: " + str(end) + " seconds")
                print(decodeAction(state, action))
                print('\n')
                ps = pstats.Stats(pr).strip_dirs().sort_stats('tottime')
                ps.print_stats(10)
            else:
                print("No valid moves")




