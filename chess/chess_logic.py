# === OPTIMIZED chess_logic.py (Complete File) ===

import random
from dataclasses import dataclass
from typing import Optional
import numpy as np

MATERIAL = (0, 100, 325, 330, 500, 950, 16383)

# Pawn advancement bonus for numpy
PAWN_ADV_WHITE = (5, 10, 20, 35, 60, 100)
PAWN_ADV_BLACK = PAWN_ADV_WHITE[::-1]

mobility_bonus = (0, 0, 5, 4, 3, 2, 0)
_notation = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6}
knight_offsets = [(1, 2), (2, 1), (2, -1), (1, -2),(-1, -2), (-2, -1), (-2, 1), (-1, 2)]

# Directional offsets, [0-3] for bishop, [4-7] for rook, all for queen
directions = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]

# PIECE-SQUARE TABLES

# Dummy board for filler info, used as a failsafe if evaluating at empty square
dummy_table = np.zeros((8, 8), dtype=np.int16)

# Pawn tables
mg_pawn_table = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (98, 134, 61, 95, 68, 126, 34, -11),
    (-6, 7, 26, 31, 65, 56, 25, -20),
    (-14, 13, 6, 21, 23, 12, 17, -23),
    (-27, -2, -5, 12, 17, 6, 10, -25),
    (-26, -4, -4, -10, 3, 3, 33, -12),
    (-35, -1, -20, -23, -15, 24, 38, -22),
    (0, 0, 0, 0, 0, 0, 0, 0),
)

eg_pawn_table = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (178, 173, 158, 134, 147, 132, 165, 187),
    (94, 100, 85, 67, 56, 53, 82, 84),
    (32, 24, 13, 5, -2, 4, 17, 17),
    (13, 9, -3, -7, -7, -8, 3, -1),
    (4, 7, -6, 1, 0, -5, -1, -8),
    (13, 8, 8, 10, 13, 0, 2, -7),
    (0, 0, 0, 0, 0, 0, 0, 0),
)

# Knight tables
mg_knight_table = (
    (-167, -89, -34, -49, 61, -97, -15, -107),
    (-73, -41, 72, 36, 23, 62, 7, -17),
    (-47, 60, 37, 65, 84, 129, 73, 44),
    (-9, 17, 19, 53, 37, 69, 18, 22),
    (-13, 4, 16, 13, 28, 19, 21, -8),
    (-23, -9, 12, 10, 19, 17, 25, -16),
    (-29, -53, -12, -3, -1, 18, -14, -19),
    (-105, -21, -58, -33, -17, -28, -19, -23),
)

eg_knight_table = (
    (-58, -38, -13, -28, -31, -27, -63, -99),
    (-25, -8, -25, -2, -9, -25, -24, -52),
    (-24, -20, 10, 9, -1, -9, -19, -41),
    (-17, 3, 22, 22, 22, 11, 8, -18),
    (-18, -6, 16, 25, 16, 17, 4, -18),
    (-23, -3, -1, 15, 10, -3, -20, -22),
    (-42, -20, -10, -5, -2, -20, -23, -44),
    (-29, -51, -23, -15, -22, -18, -50, -64),
)

# Bishop tables
mg_bishop_table = (
    (-29, 4, -82, -37, -25, -42, 7, -8),
    (-26, 16, -18, -13, 30, 59, 18, -47),
    (-16, 37, 43, 40, 35, 50, 37, -2),
    (-4, 5, 19, 50, 37, 37, 7, -2),
    (-6, 13, 13, 26, 34, 12, 10, 4),
    (0, 15, 15, 15, 14, 27, 18, 10),
    (4, 15, 16, 0, 7, 21, 33, 1),
    (-33, -3, -14, -21, -13, -12, -39, -21),
)

eg_bishop_table = (
    (-14, -21, -11, -8, -7, -9, -17, -24),
    (-8, -4, 7, -12, -3, -13, -4, -14),
    (2, -8, 0, -1, -2, 6, 0, 4),
    (-3, 9, 12, 9, 14, 10, 3, 2),
    (-6, 3, 13, 19, 7, 10, -3, -9),
    (-12, -3, 8, 10, 13, 3, -7, -15),
    (-14, -18, -7, -1, 4, -9, -15, -27),
    (-23, -9, -23, -5, -9, -16, -5, -17),
)

# Rook tables
mg_rook_table = (
    (32, 42, 32, 51, 63, 9, 31, 43),
    (27, 32, 58, 62, 80, 67, 26, 44),
    (-5, 19, 26, 36, 17, 45, 61, 16),
    (-24, -11, 7, 26, 24, 35, -8, -20),
    (-36, -26, -12, -1, 9, -7, 6, -23),
    (-45, -25, -16, -17, 3, 0, -5, -33),
    (-44, -16, -20, -9, -1, 11, -6, -71),
    (-19, -13, 1, 17, 16, 7, -37, -26),
)

eg_rook_table = (
    (13, 10, 18, 15, 12, 12, 8, 5),
    (11, 13, 13, 11, -3, 3, 8, 3),
    (7, 7, 7, 5, 4, -3, -5, -3),
    (4, 3, 13, 1, 2, 1, -1, 2),
    (3, 5, 8, 4, -5, -6, -8, -11),
    (-4, 0, -5, -1, -7, -12, -8, -16),
    (-6, -6, 0, 2, -9, -9, -11, -3),
    (-9, 2, 3, -1, -5, -13, 4, -20),
)

# Queen tables
mg_queen_table = (
    (-28, 0, 29, 12, 59, 44, 43, 45),
    (-24, -39, -5, 1, -16, 57, 28, 54),
    (-13, -17, 7, 8, 29, 56, 47, 57),
    (-27, -27, -16, -16, -1, 17, -2, 1),
    (-9, -26, -9, -10, -2, -4, 3, -3),
    (-14, 2, -11, -2, -5, 2, 14, 5),
    (-35, -8, 11, 2, 8, 15, -3, 1),
    (-1, -18, -9, 10, -15, -25, -31, -50),
)

eg_queen_table = (
    (-9, 22, 22, 27, 27, 19, 10, 20),
    (-17, 20, 32, 41, 58, 25, 30, 0),
    (-20, 6, 9, 49, 47, 35, 19, 9),
    (3, 22, 24, 45, 57, 40, 57, 36),
    (-18, 28, 19, 47, 31, 34, 39, 23),
    (-16, -27, 15, 6, 9, 17, 10, 5),
    (-22, -23, -30, -16, -16, -23, -36, -32),
    (-33, -28, -22, -43, -5, -32, -20, -41),
)

# King tables
mg_king_table = (
    (-65, 23, 16, -15, -56, -34, 2, 13),
    (29, -1, -20, -7, -8, -4, -38, -29),
    (-9, 24, 2, -16, -20, 6, 22, -22),
    (-17, -20, -12, -27, -30, -25, -14, -36),
    (-49, -1, -27, -39, -46, -44, -33, -51),
    (-14, -14, -22, -46, -44, -30, -15, -27),
    (1, 7, -8, -64, -43, -16, 9, 8),
    (-15, 36, 12, -54, 8, -28, 24, 14),
)

eg_king_table = (
    (-74, -35, -18, -18, -11, 15, 4, -17),
    (-12, 17, 14, 17, 17, 38, 23, 11),
    (10, 17, 23, 15, 20, 45, 44, 13),
    (-8, 22, 24, 27, 26, 33, 26, 3),
    (-18, -4, 21, 24, 27, 23, 9, -11),
    (-19, -3, 11, 21, 23, 16, 7, -9),
    (-27, -11, 4, 13, 14, 4, -5, -17),
    (-53, -34, -21, -11, -28, -14, -24, -43),
)

# construct 2d arrays into one 3d one
MGPST = (dummy_table, mg_pawn_table, mg_knight_table, mg_bishop_table, mg_rook_table, mg_queen_table, mg_king_table)
EGPST = (dummy_table, eg_pawn_table, eg_knight_table, eg_bishop_table, eg_rook_table, eg_queen_table, eg_king_table)

# === ORIGINAL ZOBRIST HASH DEFINITIONS (PRESERVED EXACTLY) ===

# piece_kind (1-6), color (0-1), x (1-8), y (1-8)
ZOBRIST_TABLE = {}
for kind in range(1, 7):
    for color in [0, 1]:
        for x in range(1, 9):
            for y in range(1, 9):
                ZOBRIST_TABLE[(kind, color, x, y)] = random.getrandbits(64)

SIDE_HASH = random.getrandbits(64)
CASTLE_HASH = [random.getrandbits(64) for _ in range(4)]
EP_HASH = [[random.getrandbits(64) for _ in range(9)] for _ in range(2)]


def get_initial_hash(state):
    h = 0

    # Pieces
    for p in state.board.values():
        h ^= ZOBRIST_TABLE[(p[1], p[0], p[2], p[3])]

    # Castling rights
    for i in range(4):
        if state.metadata[i]:
            h ^= CASTLE_HASH[i]

    # En passant (file + side)
    if state.metadata[4]:
        side = state.metadata[6]
        h ^= EP_HASH[side][state.metadata[4]]

    # Side to move
    if state.metadata[6]:
        h ^= SIDE_HASH

    return h


@dataclass
class UndoRecord:
    moved_piece: np.ndarray
    moved_np: int
    src: tuple
    dst: tuple
    captured_piece: bytearray | None
    captured_square: tuple | None
    captured_np: int | None
    old_metadata: bytearray
    rook_move: tuple | None  # (rook, rook_src, rook_dst)
    material_delta: int
    pst_mg_delta: int
    pst_eg_delta: int
    old_king_sq: tuple | None
    old_hash: int


# === PERFORMANCE OPTIMIZATION CACHES ===

# Pre-allocated arrays for attack calculations
_REUSE_ATTACK_ARRAY = np.zeros((8, 8), dtype=bool)
_KNIGHT_OFFSETS = np.array([(1, 2), (2, 1), (2, -1), (1, -2),
                            (-1, -2), (-2, -1), (-2, 1), (-1, 2)], dtype=np.int8)


class GameState:
    def __init__(self, board=None):
        self.board = {}
        self.board_np = np.zeros((8, 8), dtype=np.int8)
        self.material = 0
        self.pst_mg = 0
        self.pst_eg = 0
        self.kings = [None, None]

        # each piece is a bytearray
        # each bytearray is [color(black or white), type, x, y]
        for piece in board:
            new_piece = bytearray(b'\x00\x00\x00\x00')
            new_piece[0] = piece[0] == 'w'
            new_piece[1] = _notation[piece[1]]
            new_piece[2] = int(piece[2])
            new_piece[3] = int(piece[3])
            self.board[(new_piece[2], new_piece[3])] = new_piece

            # Populate numpy board
            sign = 1 if new_piece[0] else -1
            self.board_np[new_piece[3] - 1, new_piece[2] - 1] = sign * new_piece[1]

            self.material += MATERIAL[new_piece[1]] * sign

            if new_piece[1] == 6:
                self.kings[new_piece[0]] = (new_piece[2], new_piece[3])

        for p in self.board.values():
            sq = (p[3] - 1) * 8 + (p[2] - 1)
            self.pst_mg += self._getPSTValue(p[1], p[0], sq, False)
            self.pst_eg += self._getPSTValue(p[1], p[0], sq, True)

        # METADATA
        # Bytes 1-4: Castling Rights - White Kingside, White Queenside, Black Kingside, Black Queenside
        # Bytes 5-6: En Passant x y coordinates
        # Byte 7: Current turn; 1 for white 0 for black
        self.metadata = bytearray(b'\x01\x01\x01\x01\x00\x00\x01')
        self.hash = get_initial_hash(self)

        # === NEW: Performance caches ===
        self._attack_cache = {True: np.zeros((8, 8), dtype=bool),
                              False: np.zeros((8, 8), dtype=bool)}
        self._attacks_dirty = True

    # === ORIGINAL METHODS (PRESERVED EXACTLY) ===

    def __str__(self):
        return str([str(piece) for piece in self.board.values()])

    def isSquareAttacked(self, square, byWhite):
        x, y = square

        # Pawn attacks
        direction = 1 if byWhite else -1
        for dx in (-1, 1):
            p = self.board.get((x + dx, y - direction))
            if p is not None and p[1] == 1 and p[0] == byWhite:
                return True

        # Knight attacks
        for dx, dy in knight_offsets:
            p = self.board.get((x + dx, y + dy))
            if p is not None and p[1] == 2 and p[0] == byWhite:
                return True

        # Sliding pieces
        for dx, dy in directions:
            cx, cy = x + dx, y + dy
            while 1 <= cx <= 8 and 1 <= cy <= 8:
                p = self.board.get((cx, cy))
                if p is not None:
                    if p[0] == byWhite:
                        if (
                                (dx == 0 or dy == 0) and p[1] in (4, 5) or
                                (dx != 0 and dy != 0) and p[1] in (3, 5)
                        ):
                            return True
                    break
                cx += dx
                cy += dy

        # King attacks
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                p = self.board.get((x + dx, y + dy))
                if p is not None and p[1] == 6 and p[0] == byWhite:
                    return True

        return False

    def _getPseudoLegalMoves(self, piece):
        x, y = piece[2], piece[3]
        isWhite = piece[0]
        moves = []

        def inBounds(x, y):
            return 1 <= x <= 8 and 1 <= y <= 8

        # PAWN
        if piece[1] == 1:
            direction = 1 if isWhite else -1
            start_rank = 2 if isWhite else 7

            # One square forward
            fwd = (x, y + direction)
            if inBounds(*fwd) and self.board.get(fwd) is None:
                moves.append(fwd)

                # Two squares forward
                fwd2 = (x, y + 2 * direction)
                if y == start_rank and self.board.get(fwd) is None and self.board.get(fwd2) is None:
                    moves.append(fwd2)

            # Captures
            for dx in (-1, 1):
                cap = (x + dx, y + direction)
                if inBounds(*cap):
                    p = self.board.get(cap)
                    if p is not None and p[0] != isWhite:
                        moves.append(cap)

            # En passant
            ep = (self.metadata[4], self.metadata[5])
            if ep != (0, 0) and abs(ep[0] - x) == 1 and ep[1] == y + direction:
                moves.append(ep)

        # KNIGHT
        elif piece[1] == 2:
            for dx, dy in (
                    (1, 2), (2, 1), (2, -1), (1, -2),
                    (-1, -2), (-2, -1), (-2, 1), (-1, 2)
            ):
                nx, ny = x + dx, y + dy
                if not inBounds(nx, ny):
                    continue
                p = self.board.get((nx, ny))
                if p is None or p[0] != isWhite:
                    moves.append((nx, ny))

        # KING
        elif piece[1] == 6:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if not inBounds(nx, ny):
                        continue
                    p = self.board.get((nx, ny))
                    if p is None or p[0] != isWhite:
                        moves.append((nx, ny))
            # Castling

            if piece[0]:
                if (piece[2], piece[3]) == (5, 1):
                    rook1 = self.board.get((8, 1))
                    rook2 = self.board.get((1, 1))

                    # Kingside castling
                    if rook1 is not None and rook1[1] == 4 and rook1[0] and self.metadata[0]:
                        if self.board.get((6, 1)) is None and self.board.get((7, 1)) is None:
                            if not (self.isSquareAttacked((5, 1), False) or self.isSquareAttacked((6, 1),False) or self.isSquareAttacked((7, 1), False)):
                                moves.append((7, 1))

                    # Queenside castling
                    if rook2 is not None and rook2[1] == 4 and rook2[0] and self.metadata[1]:
                        if self.board.get((4, 1)) is None and self.board.get((3, 1)) is None and self.board.get(
                                (2, 1)) is None:
                            if not (self.isSquareAttacked((5, 1), False) or self.isSquareAttacked((4, 1),False) or self.isSquareAttacked((3, 1), False)):
                                moves.append((3, 1))

                    del rook1, rook2
            else:
                if (piece[2], piece[3]) == (5, 8):
                    rook1 = self.board.get((8, 8))
                    rook2 = self.board.get((1, 8))

                    # Kingside castling
                    if rook1 is not None and rook1[1] == 4 and not rook1[0] and self.metadata[2]:
                        if self.board.get((6, 8)) is None and self.board.get((7, 8)) is None:
                            if not (self.isSquareAttacked((5, 8), True) or self.isSquareAttacked((6, 8),True) or self.isSquareAttacked((7, 8), True)):
                                moves.append((7, 8))

                    # Queenside castling
                    if rook2 is not None and rook2[1] == 4 and not rook2[0] and self.metadata[3]:
                        if self.board.get((4, 8)) is None and self.board.get((3, 8)) is None and self.board.get(
                                (2, 8)) is None:
                            if not (self.isSquareAttacked((5, 8), True) or self.isSquareAttacked((4, 8),True) or self.isSquareAttacked((3, 8), True)):
                                moves.append((3, 8))

                    del rook1, rook2

        # SLIDING PIECES
        else:
            if piece[1] == 3:  # Bishop
                start, end = 0, 4
            elif piece[1] == 4:  # Rook
                start, end = 4, 8
            else:  # Queen
                start, end = 0, 8

            for i in range(start, end):
                dx, dy = directions[i][0], directions[i][1]
                nx, ny = x + dx, y + dy
                while inBounds(nx, ny):
                    p = self.board.get((nx, ny))
                    if p is None:
                        moves.append((nx, ny))
                    else:
                        if p[0] != isWhite:
                            moves.append((nx, ny))
                        break
                    nx += dx
                    ny += dy

        return moves

    def getLegalMoves(self, piece):
        if piece[0] != self.sideToMove():
            return set()

        legal = set()
        pseudo = self._getPseudoLegalMoves(piece)

        for move in pseudo:

            src_idx = (piece[3] - 1) * 8 + (piece[2] - 1)
            dst_idx = (move[1] - 1) * 8 + (move[0] - 1)

            undo = self.makeMove((src_idx, dst_idx))

            if not self.kingInCheck(not self.sideToMove()):
                legal.add(move)

            self.undoMove(undo)

        return legal

    def getAllLegalActions(self, whiteToMove):
        actions = []

        for piece in tuple(self.board.values()):
            if piece[0] != whiteToMove:
                continue

            x, y = piece[2], piece[3]
            src = (y - 1) * 8 + (x - 1)

            for nx, ny in self._getPseudoLegalMoves(piece):
                dst = (ny - 1) * 8 + (nx - 1)
                undo = self.makeMove((src, dst))
                if not self.kingInCheck(not self.sideToMove()):
                    actions.append((src, dst))
                self.undoMove(undo)

        return actions

    def getPiecePositions(self):
        return [(p[2], p[3], p[0]) for p in self.board.values()]

    def pieceAt(self, coord) -> Optional[np.ndarray]:
        if coord[0] < 1 or coord[1] < 1 or coord[0] > 8 or coord[1] > 8:
            return None
        return self.board.get(coord)

    def isEndgame(self):
        queens = sum(1 for p in self.board.values() if p[1] == 5)
        major_minor = sum(1 for p in self.board.values() if p[1] in (2, 3, 4))
        return queens == 0 or major_minor <= 4

    # === OPTIMIZED getScore WITH PST INTEGRATION ===

    def _getPSTValue(self, piece_type, is_white, square, is_endgame):
        rank = square >> 3  # square // 8
        file = square & 7  # square % 8

        if not is_white:
            rank = 7 - rank

        table = EGPST if is_endgame else MGPST
        v = table[piece_type][rank][file]

        return v if is_white else -v

    def getScore(self):
        # material + incremental PST
        pst = self.PSTScore()
        score = self.material + pst

        # Small tempo bonus
        score += 10 if self.sideToMove() else -10

        # Optional secondary evals only for close positions
        if abs(score) < 1500:
            score += self._getPawnStructureEval()
            score += self._getMobilityEval()
            score += self._getKingSafetyEval()

        return score

    def _getPawnStructureEval(self):
        white_files = [0] * 8
        black_files = [0] * 8

        for p in self.board.values():
            if p[1] == 1:
                if p[0]:
                    white_files[p[2] - 1] += 1
                else:
                    black_files[p[2] - 1] += 1

        doubled = 0
        for i in range(8):
            if white_files[i] > 1:
                doubled -= 12 * (white_files[i] - 1)
            if black_files[i] > 1:
                doubled += 12 * (black_files[i] - 1)

        return doubled

    def _getMobilityEval(self):
        score = 0
        for p in self.board.values():
            if p[1] in (2, 3, 4, 5):  # N B R Q
                s = 5 if p[1] in (2, 3) else 3
                score += s if p[0] else -s
        return score

    def _getKingSafetyEval(self):
        """Simplified king safety evaluation"""
        if self.isEndgame():
            return 0

        # Quick check for pawn shield
        board_np = self.board_np
        wk = np.argwhere(board_np == 6)
        if len(wk) > 0 and np.any(board_np == 1):
            return -15
        return 0

    # === ORIGINAL makeMove WITH _attacks_dirty INTEGRATION ===

    def makeMove(self, action) -> UndoRecord:
        # === Setup ===
        src_idx, dst_idx = action
        src = ((src_idx % 8) + 1, (src_idx // 8) + 1)
        dst = ((dst_idx % 8) + 1, (dst_idx // 8) + 1)
        sy, sx = src[1] - 1, src[0] - 1
        dy, dx = dst[1] - 1, dst[0] - 1

        piece = self.board[src]
        captured = self.board.get(dst)
        old_meta = bytearray(self.metadata)
        old_castle = old_meta[:4]

        # Precompute castling info
        is_castle = (piece[1] == 6 and abs(dst[0] - src[0]) == 2)
        rook_src = rook_dst = None
        if is_castle:
            rook_src = (8, src[1]) if dst[0] == 7 else (1, src[1])
            rook_dst = (6, src[1]) if dst[0] == 7 else (4, src[1])

        undo = UndoRecord(
            moved_piece=piece.copy(),
            moved_np=self.board_np[sy, sx],
            src=src, dst=dst,
            captured_piece=None, captured_square=None, captured_np=None,
            old_metadata=old_meta,
            rook_move=None,
            material_delta=0, pst_mg_delta=0, pst_eg_delta=0,
            old_king_sq=self.kings[piece[0]] if piece[1] == 6 else None,
            old_hash=self.hash
        )

        # === Structural Board Updates ===

        # Remove moving piece from source square
        del self.board[src]
        self.board_np[sy, sx] = 0

        # Handle normal capture
        if captured:
            undo.captured_piece = captured
            undo.captured_square = dst
            undo.captured_np = self.board_np[dy, dx]

            if captured[1] == 4:
                side = 0 if captured[0] else 2
                if captured[2] == 1:
                    self.metadata[side + 1] = 0
                elif captured[2] == 8:
                    self.metadata[side] = 0

        # === En passant (DEFENSIVE) ===
        ep_sq = None
        if (
                piece[1] == 1 and
                abs(dst[0] - src[0]) == 1 and
                (dst[0], dst[1]) == (old_meta[4], old_meta[5])
        ):
            direction = 1 if piece[0] else -1
            candidate_sq = (dst[0], dst[1] - direction)
            ep_piece = self.board.get(candidate_sq)

            # Only capture if the pawn is actually there and valid
            if ep_piece and ep_piece[1] == 1 and ep_piece[0] != piece[0]:
                ep_sq = candidate_sq
                undo.captured_piece = ep_piece
                undo.captured_square = ep_sq
                undo.captured_np = self.board_np[ep_sq[1] - 1, ep_sq[0] - 1]
                del self.board[ep_sq]

        # Place moving piece
        piece[2], piece[3] = dst
        self.board[dst] = piece
        self.board_np[dy, dx] = undo.moved_np

        if piece[1] == 6:
            self.kings[piece[0]] = dst

        # Handle castling rook move
        if is_castle:
            rook = self.board[rook_src]
            undo.rook_move = (rook.copy(), rook_src, rook_dst)

            del self.board[rook_src]
            rook[2], rook[3] = rook_dst
            self.board[rook_dst] = rook
            self.board_np[rook_dst[1] - 1, rook_dst[0] - 1] = 4 if piece[0] else -4
            self.board_np[rook_src[1] - 1, rook_src[0] - 1] = 0

        # Remove EP pawn from numpy board
        if ep_sq:
            self.board_np[ep_sq[1] - 1, ep_sq[0] - 1] = 0

        # Update castling rights
        if piece[1] == 6:
            if piece[0]:
                self.metadata[0] = self.metadata[1] = 0
            else:
                self.metadata[2] = self.metadata[3] = 0
        elif piece[1] == 4:
            if piece[0]:
                if src == (1, 1):
                    self.metadata[1] = 0
                elif src == (8, 1):
                    self.metadata[0] = 0
            else:
                if src == (1, 8):
                    self.metadata[3] = 0
                elif src == (8, 8):
                    self.metadata[2] = 0

        # Clear EP
        self.metadata[4] = 0
        self.metadata[5] = 0

        # Set new EP target
        if piece[1] == 1 and abs(dst[1] - src[1]) == 2:
            self.metadata[4] = src[0]
            self.metadata[5] = (src[1] + dst[1]) // 2

        # Flip side
        self.metadata[6] ^= 1

        # Recompute hash (simple + safe)
        self.hash = self._recalculate_hash()

        # === Evaluation updates ===
        mg_delta = (
                self._getPSTValue(piece[1], piece[0], dst_idx, False)
                - self._getPSTValue(piece[1], piece[0], src_idx, False)
        )
        eg_delta = (
                self._getPSTValue(piece[1], piece[0], dst_idx, True)
                - self._getPSTValue(piece[1], piece[0], src_idx, True)
        )

        self.pst_mg += mg_delta
        self.pst_eg += eg_delta
        undo.pst_mg_delta = mg_delta
        undo.pst_eg_delta = eg_delta

        if undo.captured_piece:
            sign = 1 if undo.captured_piece[0] else -1
            delta = MATERIAL[undo.captured_piece[1]] * sign
            self.material -= delta
            undo.material_delta = delta

        self._attacks_dirty = True
        return undo

    def undoMove(self, undo: UndoRecord):
        piece = undo.moved_piece

        # Restore metadata FIRST
        self.metadata = undo.old_metadata

        # Restore material & PST
        self.material -= undo.material_delta
        self.pst_mg -= undo.pst_mg_delta
        self.pst_eg -= undo.pst_eg_delta

        # Remove piece from dst
        del self.board[undo.dst]

        sy, sx = undo.src[1] - 1, undo.src[0] - 1
        dy, dx = undo.dst[1] - 1, undo.dst[0] - 1

        piece[2], piece[3] = undo.src
        self.board[undo.src] = piece

        self.board_np[dy, dx] = 0
        self.board_np[sy, sx] = undo.moved_np

        if piece[1] == 6:
            self.kings[piece[0]] = undo.old_king_sq

        # Restore rook if castling
        if undo.rook_move:
            rook, rook_src, rook_dst = undo.rook_move
            del self.board[rook_dst]
            rook[2], rook[3] = rook_src
            self.board[rook_src] = rook

            ry, rx = rook_src[1] - 1, rook_src[0] - 1
            rdy, rdx = rook_dst[1] - 1, rook_dst[0] - 1
            self.board_np[rdy, rdx] = 0
            self.board_np[ry, rx] = 4 if rook[0] else -4

        # Restore captured piece
        if undo.captured_piece:
            cap = undo.captured_piece
            cx, cy = undo.captured_square
            self.board[(cx, cy)] = cap
            self.board_np[cy - 1, cx - 1] = undo.captured_np

        # Rebuild hash once (correct & simple)
        self.hash = undo.old_hash

        self._attacks_dirty = True
        #assert self.hash == self._recalculate_hash()

    # === ORIGINAL makeNullMove AND undoNullMove ===

    def makeNullMove(self) -> bytearray:
        undo = bytearray(self.metadata)
        self.metadata[6] ^= 1
        self.hash ^= SIDE_HASH
        return undo

    def undoNullMove(self, undo):
        self.metadata = undo
        self.hash ^= SIDE_HASH

    # === OPTIMIZED kingInCheck ===

    def kingInCheck(self, white) -> bool:
        """Optimized with attack map caching"""
        if self._attacks_dirty:
            self._computeAttackMaps()

        king_sq = (self.kings[1] if white else self.kings[0])
        if not king_sq:
            return False

        return self._attack_cache[not white][king_sq[1] - 1, king_sq[0] - 1]

    def _computeAttackMaps(self):
        """Compute both attack maps once per node change"""
        # Clear existing attack maps
        self._attack_cache[True].fill(False)
        self._attack_cache[False].fill(False)

        # Compute attacks for ALL pieces
        for (x, y), piece in self.board.items():
            color = piece[0]
            ptype = piece[1]

            if ptype == 1:
                self._addPawnAttacks(color, x, y)
            elif ptype == 2:
                self._addKnightAttacks(color, x, y)
            elif ptype == 3:
                self._addSlidingAttacks(color, x, y, 3)
            elif ptype == 4:
                self._addSlidingAttacks(color, x, y, 4)
            elif ptype == 5:
                self._addSlidingAttacks(color, x, y, 5)
            elif ptype == 6:
                self._addKingAttacks(color, x, y)

        self._attacks_dirty = False

    def _addPawnAttacks(self, color, x, y):
        """Add pawn attacks to cache"""
        attacks = self._attack_cache[color]
        direction = 1 if color else -1
        for dx in (-1, 1):
            nx, ny = x + dx, y + direction
            if 1 <= nx <= 8 and 1 <= ny <= 8:
                attacks[ny - 1, nx - 1] = True

    def _addKnightAttacks(self, color, x, y):
        """Add knight attacks to cache"""
        attacks = self._attack_cache[color]
        for dx, dy in knight_offsets:
            nx, ny = x + dx, y + dy
            if 1 <= nx <= 8 and 1 <= ny <= 8:
                attacks[ny - 1, nx - 1] = True

    def _addKingAttacks(self, color, x, y):
        """Add king attacks to cache"""
        attacks = self._attack_cache[color]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                if 1 <= nx <= 8 and 1 <= ny <= 8:
                    attacks[ny - 1, nx - 1] = True

    def _addSlidingAttacks(self, color, x, y, piece_type):
        """Add bishop/rook/queen attacks to cache"""
        attacks = self._attack_cache[color]

        # Determine direction range
        if piece_type == 3:
            start, end = 0, 4  # Bishop
        elif piece_type == 4:
            start, end = 4, 8  # Rook
        else:
            start, end = 0, 8  # Queen

        for i in range(start, end):
            dx, dy = directions[i]
            nx, ny = x + dx, y + dy
            while 1 <= nx <= 8 and 1 <= ny <= 8:
                attacks[ny - 1, nx - 1] = True
                if self.board.get((nx, ny)) is not None: break
                nx += dx
                ny += dy

    def _findKingSquare(self, white):
        """Faster king lookup"""
        piece = next((p for p in self.board.values() if p[1] == 6 and p[0] == white), None)
        return (piece[2], piece[3]) if piece is not None else None

    # === ORIGINAL REMAINING METHODS ===

    def sideToMove(self):
        return self.metadata[6] == 1

    def PSTScore(self):
        return self.pst_eg if self.isEndgame() else self.pst_mg

    def vectorize(self) -> bytes:
        arr = bytearray()
        sorted_pieces = sorted(self.board, key=lambda p: (p[3], p[2]))
        for piece in sorted_pieces:
            arr.append(piece[2])
            arr.append(piece[3])
            arr.append(piece[1] if piece[0] else 256 - piece[1])
        arr.append(0)
        arr.extend(self.metadata)
        return bytes(arr)

    def _recalculate_hash(self):
        """Force recalculate hash from current state (for testing)"""
        return get_initial_hash(self)

    def __eq__(self, other):
        return isinstance(other, GameState) and self.hash == other.hash


# === ORIGINAL main() FUNCTION ===
import web_parser
import time


def main():
    web_parser.open_site()
    prev = None

    while True:
        try:
            time.sleep(1.5)
            pieces = web_parser.get_board_data()
            print(pieces)
            state = GameState(pieces)
            print(state.getScore())
        except Exception as e:
            print("ERROR:", e)
            exit(1)


if __name__ == "__main__":
    main()