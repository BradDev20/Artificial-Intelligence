import math, random
from dataclasses import dataclass
from typing import Optional
import numpy as np

import web_parser
import time, itertools

#Tuple values are the name, centipawn value, and value per square it can travel to
piece_values = {1: ("Pawn", 100, 0), 2: ("Knight", 325, 5), 3: ("Bishop", 330, 4),
          4: ("Rook", 500, 3), 5: ("Queen", 950, 2), 6: ("King", 0, 0)}
_notation = {'p': 1, 'n': 2, 'b':3, 'r': 4, 'q': 5, 'k': 6}

# Material values for numpy
MATERIAL = np.array([0, 100, 325, 330, 500, 950, 0], dtype=np.int16)

# Pawn advancement bonus for numpy
PAWN_ADV_WHITE = np.array([0, 0, 5, 10, 20, 35, 60, 100], dtype=np.int16)
PAWN_ADV_BLACK = PAWN_ADV_WHITE[::-1]

# piece_kind (1-6), color (0-1), x (1-8), y (1-8)
ZOBRIST_TABLE = {}
for kind in range(1, 7):
    for color in [0, 1]:
        for x in range(1, 9):
            for y in range(1, 9):
                ZOBRIST_TABLE[(kind, color, x, y)] = random.getrandbits(64)

# Hash for side to move and metadata (castling/ep)
SIDE_HASH = random.getrandbits(64)
META_HASH = [random.getrandbits(64) for _ in range(16)] # Bits for metadata combos

def get_initial_hash(state):
    h = 0
    for p in state.board.values():
        h ^= ZOBRIST_TABLE[(p[1], int(p[0]), p[2], p[3])]
    if state.metadata[6] == 1: # Black to move
        h ^= SIDE_HASH
    # Simplified: XOR in metadata bytes
    for i, val in enumerate(state.metadata):
        h ^= hash((i, val))
    return h


@dataclass
class UndoRecord:
    moved_piece: bytearray
    src: tuple
    dst: tuple
    captured_piece: bytearray | None
    captured_square: tuple | None
    captured_np: int | None
    old_metadata: bytearray
    rook_move: tuple | None  # (rook, rook_src, rook_dst)


class GameState:
    def __init__(self, board=None):
        self.board = {}
        self.board_np = np.zeros((8, 8), dtype=np.int8)

        # each piece is a bytearray
        # each bytearray is [color(black or white), type, x, y]
        for piece in board:
            new_piece = bytearray(b'\x00\x00\x00\x00')
            new_piece[0] = piece[0] == 'w'
            new_piece[1] = _notation[piece[1]]
            new_piece[2] = int(piece[2])
            new_piece[3] = int(piece[3])
            self.board[(new_piece[2],new_piece[3])] = new_piece

            # Populate numpy board
            sign = 1 if new_piece[0] else -1
            self.board_np[new_piece[2] - 1, new_piece[3] - 1] = sign * new_piece[1]

        #METADATA
        #Bytes 1-4: Castling Rights - White Kingside, White Queenside, Black Kingside, Black Queenside
        #Bytes 5-6: En Passant x y coordinates
        #Byte 7: Current turn; 1 for white 0 for black
        self.metadata = bytearray(b'\x01\x01\x01\x01\x00\x00\x01')

        self.hash = get_initial_hash(self)

    def __str__(self):
        return str([str(piece) for piece in self.board.values()])

    def isSquareAttacked(self, square, byWhite):
        x, y = square

        # ----------------
        # Pawn attacks
        # ----------------
        direction = 1 if byWhite else -1
        for dx in (-1, 1):
            p = self.pieceAt((x + dx, y - direction))
            if p and p[1] == 1 and p[0] == byWhite:
                return True

        # ----------------
        # Knight attacks
        # ----------------
        knight_offsets = [
            (1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2)
        ]
        for dx, dy in knight_offsets:
            p = self.pieceAt((x + dx, y + dy))
            if p and p[1] == 2 and p[0] == byWhite:
                return True

        # ----------------
        # Sliding pieces
        # ----------------
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # rook
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # bishop
        ]

        for dx, dy in directions:
            cx, cy = x + dx, y + dy
            while 1 <= cx <= 8 and 1 <= cy <= 8:
                p = self.pieceAt((cx, cy))
                if p:
                    if p[0] == byWhite:
                        if (
                                (dx == 0 or dy == 0) and p[1] in (4, 5) or
                                (dx != 0 and dy != 0) and p[1] in (3, 5)
                        ):
                            return True
                    break
                cx += dx
                cy += dy

        # ----------------
        # King attacks
        # ----------------
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                p = self.pieceAt((x + dx, y + dy))
                if p and p[1] == 6 and p[0] == byWhite:
                    return True

        return False

    def _getPseudoLegalMoves(self, piece):
        x, y = piece[2], piece[3]
        isWhite = piece[0]
        moves = []

        def inBounds(x, y):
            return 1 <= x <= 8 and 1 <= y <= 8

        # --------------------
        # PAWN
        # --------------------
        if piece[1] == 1:
            direction = 1 if isWhite else -1
            start_rank = 2 if isWhite else 7

            # One square forward
            fwd = (x, y + direction)
            if inBounds(*fwd) and not self.pieceAt(fwd):
                moves.append(fwd)

                # Two squares forward
                fwd2 = (x, y + 2 * direction)
                if y == start_rank and not self.pieceAt(fwd) and not self.pieceAt(fwd2):
                    moves.append(fwd2)

            # Captures
            for dx in (-1, 1):
                cap = (x + dx, y + direction)
                if inBounds(*cap):
                    p = self.pieceAt(cap)
                    if p and p[0] != isWhite:
                        moves.append(cap)

            # En passant
            ep = (self.metadata[4], self.metadata[5])
            if ep != (0, 0) and abs(ep[0] - x) == 1 and ep[1] == y + direction:
                moves.append(ep)

        # --------------------
        # KNIGHT
        # --------------------
        elif piece[1] == 2:
            for dx, dy in (
                    (1, 2), (2, 1), (2, -1), (1, -2),
                    (-1, -2), (-2, -1), (-2, 1), (-1, 2)
            ):
                nx, ny = x + dx, y + dy
                if not inBounds(nx, ny):
                    continue
                p = self.pieceAt((nx, ny))
                if not p or p[0] != isWhite:
                    moves.append((nx, ny))

        # --------------------
        # KING
        # --------------------
        elif piece[1] == 6:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if not inBounds(nx, ny):
                        continue
                    p = self.pieceAt((nx, ny))
                    if not p or p[0] != isWhite:
                        moves.append((nx, ny))
            # Castling

            if piece[0]:
                if (piece[2], piece[3]) == (5,1):
                    rook1 = self.pieceAt((8,1))
                    rook2 = self.pieceAt((1,1))

                    #Kingside castling
                    if rook1 and rook1[1] == 4 and rook1[0] and self.metadata[0]:
                        if not self.pieceAt((6,1)) and not self.pieceAt((7,1)):
                            if not(self.isSquareAttacked((5,1), False) or self.isSquareAttacked((6,1), False) or self.isSquareAttacked((7,1), False)):
                                moves.append((7,1))

                    #Queenside castling
                    if rook2 and rook2[1] == 4 and rook2[0] and self.metadata[1]:
                        if not self.pieceAt((4, 1)) and not self.pieceAt((3, 1)) and not self.pieceAt((2, 1)):
                            if not(self.isSquareAttacked((5,1), False) or self.isSquareAttacked((4,1), False) or self.isSquareAttacked((3,1), False)):
                                moves.append((3, 1))

                    del rook1, rook2
            else:
                if (piece[2], piece[3]) == (5,8):
                    rook1 = self.pieceAt((8,8))
                    rook2 = self.pieceAt((1,8))

                    #Kingside castling
                    if rook1 and rook1[1] == 4 and not rook1[0] and self.metadata[2]:
                        if not self.pieceAt((6, 8)) and not self.pieceAt((7, 8)):
                            if not(self.isSquareAttacked((5,8), True) or self.isSquareAttacked((6,8), True) or self.isSquareAttacked((7,8), True)):
                                moves.append((7,8))

                    #Queenside castling
                    if rook2 and rook2[1] == 4 and not rook2[0] and self.metadata[3]:
                        if not self.pieceAt((4, 8)) and not self.pieceAt((3, 8)) and not self.pieceAt((2, 8)):
                            if not(self.isSquareAttacked((5,8), True) or self.isSquareAttacked((4,8), True) or self.isSquareAttacked((3,8), True)):
                                moves.append((3, 8))

                    del rook1,rook2

        # --------------------
        # SLIDING PIECES
        # --------------------
        else:
            if piece[1] == 3:  # Bishop
                directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            elif piece[1] == 4:  # Rook
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            else:  # Queen
                directions = [
                    (1, 1), (1, -1), (-1, 1), (-1, -1),
                    (1, 0), (-1, 0), (0, 1), (0, -1)
                ]

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                while inBounds(nx, ny):
                    p = self.pieceAt((nx, ny))
                    if not p:
                        moves.append((nx, ny))
                    else:
                        if p[0] != isWhite:
                            moves.append((nx, ny))
                        break
                    nx += dx
                    ny += dy

        return moves

    def getLegalMoves(self, piece):
        # Safety: only generate moves for side to move
        if piece[0] != self.sideToMove():
            return set()

        legal = set()
        pseudo = self._getPseudoLegalMoves(piece)

        for move in pseudo:

            src_idx = (piece[3] - 1) * 8 + (piece[2] - 1)
            dst_idx = (move[1] - 1) * 8 + (move[0] - 1)

            undo = self.makeMove((src_idx, dst_idx))  # Now passing valid action

            # Legal = own king not in check
            if not self.kingInCheck(not self.sideToMove()):
                legal.add(move)

            self.undoMove(undo)

        return legal

    def getAllLegalActions(self, whiteToMove):
        """
        Returns all legal actions for the current state in a compact format.

        Each action is a tuple: (from_square, to_square)
        Squares are numbered 0–63 (a1=0, b1=1, ..., h8=63)

        :return: set of actions {(from_square, to_square), ...}
        """
        actions = []

        # snapshot board(prevents mutation during iteration)
        pieces = list(self.board.values())

        for piece in pieces:
            if piece is None:
                continue
            if piece[0] != whiteToMove:
                continue

            x, y = piece[2], piece[3]
            src_idx = (y - 1) * 8 + (x - 1)

            for (nx, ny) in self._getPseudoLegalMoves(piece):
                dst_idx = (ny - 1) * 8 + (nx - 1)
                actions.append((src_idx, dst_idx))

        return actions




    def getPiecePositions(self):
        return [(p[2],p[3],p[0]) for p in self.board.values()]

    def pieceAt(self, coord) -> Optional[bytearray]:
        if coord[0] < 1 or coord[1] < 1 or coord[0] > 8 or coord[1] > 8:
            return None
        return self.board.get(coord)

    def isEndgame(self):
        # Endgame if queens are gone OR low material
        queens = sum(1 for p in self.board.values() if p[1] == 5)
        major_minor = sum(1 for p in self.board.values() if p[1] in (2, 3, 4))
        return queens == 0 or major_minor <= 4

    def getScore(self):

        # for checking if passed pawns are blocked by adjacent files
        def adjacentBlock(mask):
            return mask | np.roll(mask, 1, axis=0)| np.roll(mask, -1, axis=0)

        # penalty for if the king isn't shielded by pawns
        def pawnShieldPenalty(pawns, kx, ky, direction):
            penalty = 0
            shield_rank = ky + direction
            if 0 <= shield_rank < 8:
                for dx in (-1, 0, 1):
                    fx = kx + dx
                    if 0 <= fx < 8 and not pawns[fx, shield_rank]:
                        penalty += 10
            return penalty

        # penalty for if there is no friendly pawn in front of the king
        def openFilePenalty(pawn_files, kx):
            penalty = 0
            for dx in (-1, 0, 1):
                fx = kx + dx
                if 0 <= fx < 8 == pawn_files[fx]:
                    penalty += 8
            return penalty

        board_np = self.board_np

        piece_ids = np.abs(board_np)
        material = MATERIAL[piece_ids]

        # initialize score to material value sum
        score = np.sum(material * np.sign(board_np))

        # -----------------------------
        # PIECE PRE-COLLECTION
        # -----------------------------
        white_pawns = board_np == 1
        black_pawns = board_np == -1
        white_rooks = board_np == 4
        black_rooks = board_np == -4

        # For king safety, these are *positions*, not ints
        white_king = np.argwhere(board_np == 6)
        black_king = np.argwhere(board_np == -6)

        # -----------------------------
        # Pawn structure (cached)
        # -----------------------------
        pawn_files_white = np.sum(white_pawns, axis=0)
        pawn_files_black = np.sum(black_pawns, axis=0)

        # -----------------------------
        # PAWN EVALUATION
        # -----------------------------

        # doubled penalty
        white_doubled = np.sum(np.maximum(pawn_files_white - 1, 0)) * 12
        black_doubled = np.sum(np.maximum(pawn_files_black - 1, 0)) * 12

        score -= white_doubled
        score += black_doubled

        # isolation penalty
        left_white = np.roll(pawn_files_white, 1)
        right_white = np.roll(pawn_files_white, -1)
        isolated_white = (pawn_files_white > 0) & (left_white == 0) & (right_white == 0)

        left_black = np.roll(pawn_files_black, 1)
        right_black = np.roll(pawn_files_black, -1)
        isolated_black = (pawn_files_black > 0) & (left_black == 0) & (right_black == 0)

        score -= np.sum(isolated_white) * 15
        score += np.sum(isolated_black) * 15

        # ---------------------------
        # ADVANCEMENT/FORWARDS VALUE
        # ---------------------------
        ranks = np.arange(8)
        white_adv = PAWN_ADV_WHITE[ranks] @ np.sum(white_pawns, axis=0)
        black_adv = PAWN_ADV_BLACK[ranks] @ np.sum(black_pawns, axis=0)

        black_ahead = np.cumsum(black_pawns[:, ::-1], axis=1)[:, ::-1]
        white_ahead = np.cumsum(white_pawns, axis=1)

        # >0 == Not passed opposing pawn
        black_block = adjacentBlock(black_ahead)
        white_block = adjacentBlock(white_ahead)

        # get all pawns that are 'passed'
        passed_white = white_pawns & (black_block == 0)
        passed_black = black_pawns & (white_block == 0)

        score += white_adv
        score -= black_adv

        w_passed = np.sum(PAWN_ADV_WHITE[ranks] * passed_white.sum(axis=0))
        b_passed = np.sum(PAWN_ADV_BLACK[ranks] * passed_black.sum(axis=0))

        # if a passed pawn is blocked, halve its bonus
        blocked_w = passed_white & (np.roll(board_np, -1, axis=1) != 0)
        w_passed -= np.sum(PAWN_ADV_WHITE[ranks] * blocked_w.sum(axis=0)) // 2

        blocked_b = passed_black & (np.roll(board_np, -1, axis=1) != 0)
        b_passed -= np.sum(PAWN_ADV_BLACK[ranks] * blocked_b.sum(axis=0)) // 2

        # Score modifier = passed pawn * value for being on its square
        score += np.sum(PAWN_ADV_WHITE[ranks] * passed_white.sum(axis=0))
        score -= np.sum(PAWN_ADV_BLACK[ranks] * passed_black.sum(axis=0))


        # --------------------------
        # CONNECTED PAWNS
        # --------------------------

        # Get shift one left and right
        white_left = np.roll(white_pawns, 1, axis=0)
        white_right = np.roll(white_pawns, -1, axis=0)

        black_left = np.roll(black_pawns, 1, axis=0)
        black_right = np.roll(black_pawns, -1, axis=0)

        # find pawns connected horizontally
        white_same_rank = white_left | white_right
        black_same_rank = black_left | black_right

        # pawns connected diagonally
        white_support = np.roll(white_left | white_right, 1, axis=1)
        black_support = np.roll(black_left | black_right, -1, axis=1)

        # connected pawn bonus mask
        connected_white = white_pawns & (white_same_rank | white_support)
        connected_black = black_pawns & (black_same_rank | black_support)

        score += np.sum((PAWN_ADV_WHITE * 0.15) * connected_white.sum(axis=0))
        score -= np.sum((PAWN_ADV_BLACK * 0.15) * connected_black.sum(axis=0))

        # -----------------
        # BACKWARD PAWNS
        # -----------------

        # pawns blocked by a piece in front of them
        white_blocked = white_pawns & (np.roll(board_np != 0, 1, axis=1))
        black_blocked = black_pawns & (np.roll(board_np != 0, -1, axis=1))

        # supportive pieces
        white_support_behind = np.roll(white_left | white_right, -1, axis=1)
        black_support_behind = np.roll(black_left | black_right, 1, axis=1)

        # a pawn is 'backwards' if it is both blocked and unsupported
        backward_white = white_blocked & ~white_support_behind
        backward_black = black_blocked & ~black_support_behind

        score -= np.sum(backward_white) * 12
        score += np.sum(backward_black) * 12

        # -----------------------------
        # Rook evaluation (open / semi-open files)
        # -----------------------------
        white_rook_files = np.any(white_rooks, axis=1)
        black_rook_files = np.any(black_rooks, axis=1)

        # +15 for own file empty, +7 for enemy and vice versa for black
        # White rooks
        score += np.sum(white_rook_files & (pawn_files_white == 0)) * 15
        score += np.sum(white_rook_files & (pawn_files_black == 0)) * 7

        # Black rooks
        score -= np.sum(black_rook_files & (pawn_files_black == 0)) * 15
        score -= np.sum(black_rook_files & (pawn_files_white == 0)) * 7

        # -----------------------------
        # KING SAFETY
        # -----------------------------

        #only perform the safety checks if the king exists
        if len(white_king):
            wx, wy = white_king[0]
            score -= pawnShieldPenalty(white_pawns, wx, wy, 1)
            score -= openFilePenalty(pawn_files_white, wx)

        if len(black_king):
            bx, by = black_king[0]
            score += pawnShieldPenalty(black_pawns, bx, by, -1)
            score += openFilePenalty(pawn_files_black, bx)

        return score

    def makeMove(self, action) -> UndoRecord:
        """
        Makes a move on the board with the given action. This modifies the board state.
        :param action: A tuple (int, int) representing the action to perform
        :return: A record of the board's info before the move, used for undoing moves later
        """
        src = ((action[0] % 8) + 1, (action[0] // 8) + 1)
        dst = ((action[1] % 8) + 1, (action[1] // 8) + 1)

        #used for numpy
        sx = src[0] - 1
        sy = src[1] - 1
        dx = dst[0] - 1
        dy = dst[1] - 1

        piece = self.pieceAt(src)
        captured = self.pieceAt(dst)
        captured_np = self.board_np[dx, dy]

        piece_code = self.board_np[sx, sy]
        self.board_np[sx, sy] = 0
        self.board_np[dx, dy] = piece_code

        undo = UndoRecord(
            moved_piece=piece,
            src=src,
            dst=dst,
            captured_piece=captured,
            captured_square=dst if captured else None,
            captured_np=captured_np,
            old_metadata=bytearray(self.metadata),
            rook_move=None
        )

        # XOR moving piece from old pos
        self.hash ^= ZOBRIST_TABLE[(piece[1], piece[0], src[0], src[1])]

        # remove captured piece
        if captured:

            # XOR captured piece
            self.hash ^= ZOBRIST_TABLE[(captured[1], captured[0], dst[0], dst[1])]
            del self.board[dst]

        # en passant capture
        if piece[1] == 1 and dst == (self.metadata[4], self.metadata[5]):
            cap_sq = (dst[0], src[1])
            ep_piece = self.board.get(cap_sq)

            # validate en-passant
            if ep_piece and ep_piece[1] == 1 and ep_piece[0] != piece[0]:
                undo.captured_piece = ep_piece
                undo.captured_square = cap_sq

                # XOR out captured pawn
                self.hash ^= ZOBRIST_TABLE[(ep_piece[1], ep_piece[0], cap_sq[0], cap_sq[1])]

                del self.board[cap_sq]

        # move piece
        del self.board[src]
        piece[2], piece[3] = dst
        self.board[dst] = piece

        # XOR moving the piece
        self.hash ^= ZOBRIST_TABLE[(piece[1], piece[0], dst[0], dst[1])]

        # castling
        if piece[1] == 6 and abs(dst[0] - src[0]) == 2:
            if dst[0] == 7:  # kingside
                rook_src = (8, src[1])
                rook_dst = (6, src[1])
            else:  # queenside
                rook_src = (1, src[1])
                rook_dst = (4, src[1])

            rook = self.pieceAt(rook_src)
            if rook:
                # XOR out old rook and XOR in new one
                self.hash ^= ZOBRIST_TABLE[(rook[1], rook[0], rook_src[0], rook_src[1])]
                del self.board[rook_src]
                rook[2], rook[3] = rook_dst
                self.board[rook_dst] = rook
                self.hash ^= ZOBRIST_TABLE[(rook[1], rook[0], rook_dst[0], rook_dst[1])]
                undo.rook_move = (rook, rook_src, rook_dst)

        # update metadata
        self.metadata[4] = 0
        self.metadata[5] = 0

        if piece[1] == 1 and abs(dst[1] - src[1]) == 2:
            self.metadata[4] = src[0]
            self.metadata[5] = (src[1] + dst[1]) // 2

        self.metadata[6] = 1 - self.metadata[6]
        self.hash ^= SIDE_HASH

        return undo

    def undoMove(self, undo: UndoRecord):
        """
        Undoes the move with the given record.
        :param undo: an UndoRecord instance containing the previous board state
        """
        piece = undo.moved_piece

        # XOR out the piece from the destination (where it is now)
        self.hash ^= ZOBRIST_TABLE[(piece[1], piece[0], undo.dst[0], undo.dst[1])]

        if undo.dst in self.board:
            del self.board[undo.dst]

        piece[2], piece[3] = undo.src
        self.board[undo.src] = piece

        # XOR the piece back into its source position
        self.hash ^= ZOBRIST_TABLE[(piece[1], piece[0], undo.src[0], undo.src[1])]

        if undo.captured_piece:
            cap_p = undo.captured_piece
            cap_sq = undo.captured_square
            self.board[cap_sq] = cap_p
            # XOR the captured piece back onto the board
            self.hash ^= ZOBRIST_TABLE[(cap_p[1], cap_p[0], cap_sq[0], cap_sq[1])]

        if undo.rook_move:
            rook, r_src, r_dst = undo.rook_move
            # XOR out rook from dst, XOR back into src
            self.hash ^= ZOBRIST_TABLE[(rook[1], rook[0], r_dst[0], r_dst[1])]
            del self.board[r_dst]
            rook[2], rook[3] = r_src
            self.board[r_src] = rook
            self.hash ^= ZOBRIST_TABLE[(rook[1], rook[0], r_src[0], r_src[1])]

        # Restore NumPy array
        sx = undo.src[0] - 1
        sy = undo.src[1] - 1
        dx = undo.dst[0] - 1
        dy = undo.dst[1] - 1

        # Put piece back to source
        piece_code = self.board_np[dy, dx]  # it should still be there
        self.board_np[dy, dx] = 0
        self.board_np[sy, sx] = piece_code

        # If there was a capture, restore captured piece code
        if undo.captured_np:
            cap_x = undo.captured_square[0] - 1 if undo.captured_square else dx
            cap_y = undo.captured_square[1] - 1 if undo.captured_square else dy
            self.board_np[cap_y, cap_x] = undo.captured_np

        self.metadata = undo.old_metadata
        self.hash ^= SIDE_HASH

    def makeNullMove(self) -> bytearray:
        """
        Creates a "null move" for null pruning by XORing the state hash and its metadata with the Zobrist side hash.
        Does not change anything about the board except the player turn.
        :return: the state's metadata before the null move
        """
        undo = bytearray(self.metadata)
        self.metadata[6] ^= 1
        self.hash ^= SIDE_HASH
        return undo

    def undoNullMove(self, undo):
        """
        Undoes a "null move".
        :param undo: the metadata returned from calling makeNullMove
        """
        self.metadata = undo
        self.hash ^= SIDE_HASH

    def kingInCheck(self, white) -> bool:
        """
        Determines if the king is in check.
        :param white: If True, checks for the White king; otherwise, checks for the Black king.
        """
        king = next((p for p in self.board.values() if p[1] == 6 and p[0] == white), None)
        if not king:
            return False

        kx, ky = king[2], king[3]
        enemy_white = not white

        # Check for Knight attacks
        knight_moves = [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]
        for dx, dy in knight_moves:
            p = self.pieceAt((kx + dx, ky + dy))
            if p and p[1] == 2 and p[0] == enemy_white:
                return True

        # 2. Check for Rook/Queen attacks (straight) & Bishop/Queen (diagonal)
        directions = [
            (0, 1, {4, 5}), (0, -1, {4, 5}), (1, 0, {4, 5}), (-1, 0, {4, 5}),  # Rook/Queen
            (1, 1, {3, 5}), (1, -1, {3, 5}), (-1, 1, {3, 5}), (-1, -1, {3, 5})  # Bishop/Queen
        ]
        for dx, dy, attackers in directions:
            tx, ty = kx + dx, ky + dy
            while 1 <= tx <= 8 and 1 <= ty <= 8:
                p = self.pieceAt((tx, ty))
                if p:
                    if p[0] == enemy_white and p[1] in attackers:
                        return True
                    break  # Blocked by any piece
                tx += dx
                ty += dy

        # 3. Check for Pawn attacks
        pawn_y_dir = 1 if white else -1
        for dx in [-1, 1]:
            p = self.pieceAt((kx + dx, ky + pawn_y_dir))
            if p and p[1] == 1 and p[0] == enemy_white:
                return True

        # 4. Check for adjacent King (illegal position, but good for safety)
        for dx, dy in itertools.product([-1, 0, 1], repeat=2):
            if dx == 0 and dy == 0: continue
            p = self.pieceAt((kx + dx, ky + dy))
            if p and p[1] == 6 and p[0] == enemy_white:
                return True

        return False

    def sideToMove(self):
        return self.metadata[6] == 1

    def vectorize(self) -> bytes:
        """
                Converts the GameState into a byte vector.
                Format is as follows:
                [x1, y1, pieceType1, x2, y2, pieceType2,..., null byte, castlingRights, enPassantX, enPassantY, currentTurn]

                Note: In this case, if a piece is Black, its "type" is 256 - its usual. Ex. a Black knight
                is type 254.

                Vectorization is for state representation for neural networks/Q-learning.
                :return:
                """
        arr = bytearray()

        sorted_pieces = sorted(self.board, key=lambda p: (p[3], p[2]))

        for piece in sorted_pieces:
            arr.append(piece[2])
            arr.append(piece[3])
            arr.append(piece[1] if piece[0] else 256 - piece[1])

        arr.append(0)
        # Add metadata bytes
        arr.extend(self.metadata)

        return bytes(arr)

    def __eq__(self, other):
        return isinstance(other, GameState) and self.hash == other.hash


def main():

    web_parser.open_site()

    prev = None

    while True:
        try:
            time.sleep(1.5)
            pieces = web_parser.get_board_data()
            state = GameState(pieces)


            print(state.getScore())
        except Exception as e:
            print("ERROR:",e)
            exit(1)

if __name__ == "__main__":
    main()
