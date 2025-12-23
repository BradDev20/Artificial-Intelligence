import math, random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import web_parser
import time, itertools

#Tuple values are the name, centipawn value, and value per square it can travel to
piece_values = {1: ("Pawn", 100, 0), 2: ("Knight", 325, 5), 3: ("Bishop", 330, 4),
          4: ("Rook", 500, 3), 5: ("Queen", 950, 2), 6: ("King", 0, 0)}
_notation = {'p': 1, 'n': 2, 'b':3, 'r': 4, 'q': 5, 'k': 6}

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
    for p in state.board:
        h ^= ZOBRIST_TABLE[(p.data[1], int(p.data[0]), p.data[2], p.data[3])]
    if state.metadata[6] == 1: # Black to move
        h ^= SIDE_HASH
    # Simplified: XOR in metadata bytes
    for i, val in enumerate(state.metadata):
        h ^= hash((i, val))
    return h


@dataclass
class UndoRecord:
    moved_piece: object
    src: tuple
    dst: tuple
    captured_piece: object | None
    captured_square: tuple | None
    old_metadata: bytearray
    rook_move: tuple | None  # (rook, rook_src, rook_dst)


class GameState:
    def __init__(self, board=None):
        self.board = set()
        self.lookup = defaultdict(lambda: None)

        for piece in board:
            x = int(piece[2])
            y = int(piece[3])
            new_piece = Piece((piece[0] == 'w'), _notation[piece[1]], x, y)
            self.board.add(new_piece)
            self.lookup[(x,y)] = new_piece

        #METADATA
        #Bytes 1-4: Castling Rights - White Kingside, White Queenside, Black Kingside, Black Queenside
        #Bytes 5-6: En Passant x y coordinates
        #Byte 7: Current turn; 1 for white 0 for black
        self.metadata = bytearray(b'\x01\x01\x01\x01\x00\x00\x01')

        self.hash = get_initial_hash(self)

    def __str__(self):
        return str([str(piece) for piece in self.board])

    def pawnAttacksSquare(self, x, y, byWhite):
        """
        Returns True if square (x, y) is attacked by an enemy pawn
        """
        direction = -1 if byWhite else 1
        for dx in (-1, 1):
            px, py = x + dx, y + direction
            if 1 <= px <= 8 and 1 <= py <= 8:
                p = self.pieceAt((px, py))
                if p and p.data[1] == 1 and p.data[0] == byWhite:
                    return True
        return False

    def isSquareAttacked(self, square, byWhite):
        x, y = square

        # ----------------
        # Pawn attacks
        # ----------------
        direction = 1 if byWhite else -1
        for dx in (-1, 1):
            p = self.pieceAt((x + dx, y - direction))
            if p and p.data[1] == 1 and p.data[0] == byWhite:
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
            if p and p.data[1] == 2 and p.data[0] == byWhite:
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
                    if p.data[0] == byWhite:
                        if (
                                (dx == 0 or dy == 0) and p.data[1] in (4, 5) or
                                (dx != 0 and dy != 0) and p.data[1] in (3, 5)
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
                if p and p.data[1] == 6 and p.data[0] == byWhite:
                    return True

        return False

    def _getPseudoLegalMoves(self, piece):
        x, y = piece.getPosition()
        isWhite = piece.data[0]
        moves = []

        def inBounds(x, y):
            return 1 <= x <= 8 and 1 <= y <= 8

        # --------------------
        # PAWN
        # --------------------
        if piece.data[1] == 1:
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
                    if p and p.data[0] != isWhite:
                        moves.append(cap)

            # En passant
            ep = (self.metadata[4], self.metadata[5])
            if ep != (0, 0) and abs(ep[0] - x) == 1 and ep[1] == y + direction:
                moves.append(ep)

        # --------------------
        # KNIGHT
        # --------------------
        elif piece.data[1] == 2:
            for dx, dy in (
                    (1, 2), (2, 1), (2, -1), (1, -2),
                    (-1, -2), (-2, -1), (-2, 1), (-1, 2)
            ):
                nx, ny = x + dx, y + dy
                if not inBounds(nx, ny):
                    continue
                p = self.pieceAt((nx, ny))
                if not p or p.data[0] != isWhite:
                    moves.append((nx, ny))

        # --------------------
        # KING
        # --------------------
        elif piece.data[1] == 6:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if not inBounds(nx, ny):
                        continue
                    p = self.pieceAt((nx, ny))
                    if not p or p.data[0] != isWhite:
                        moves.append((nx, ny))
            # Castling

            if piece.data[0]:
                if piece.getPosition() == (5,1):
                    rook1 = self.pieceAt((8,1))
                    rook2 = self.pieceAt((1,1))

                    #Kingside castling
                    if rook1 and rook1.data[1] == 4 and rook1.data[0] and self.metadata[0]:
                        if not self.pieceAt((6,1)) and not self.pieceAt((7,1)):
                            if not(self.isSquareAttacked((5,1), False) or self.isSquareAttacked((6,1), False) or self.isSquareAttacked((7,1), False)):
                                moves.append((7,1))

                    #Queenside castling
                    if rook2 and rook2.data[1] == 4 and rook2.data[0] and self.metadata[1]:
                        if not self.pieceAt((4, 1)) and not self.pieceAt((3, 1)) and not self.pieceAt((2, 1)):
                            if not(self.isSquareAttacked((5,1), False) or self.isSquareAttacked((4,1), False) or self.isSquareAttacked((3,1), False)):
                                moves.append((3, 1))

                    del rook1, rook2
            else:
                if piece.getPosition() == (5,8):
                    rook1 = self.pieceAt((8,8))
                    rook2 = self.pieceAt((1,8))

                    #Kingside castling
                    if rook1 and rook1.data[1] == 4 and not rook1.data[0] and self.metadata[2]:
                        if not self.pieceAt((6, 8)) and not self.pieceAt((7, 8)):
                            if not(self.isSquareAttacked((5,8), True) or self.isSquareAttacked((6,8), True) or self.isSquareAttacked((7,8), True)):
                                moves.append((7,8))

                    #Queenside castling
                    if rook2 and rook2.data[1] == 4 and not rook2.data[0] and self.metadata[3]:
                        if not self.pieceAt((4, 8)) and not self.pieceAt((3, 8)) and not self.pieceAt((2, 8)):
                            if not(self.isSquareAttacked((5,8), True) or self.isSquareAttacked((4,8), True) or self.isSquareAttacked((3,8), True)):
                                moves.append((3, 8))

                    del rook1,rook2

        # --------------------
        # SLIDING PIECES
        # --------------------
        else:
            if piece.data[1] == 3:  # Bishop
                directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            elif piece.data[1] == 4:  # Rook
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
                        if p.data[0] != isWhite:
                            moves.append((nx, ny))
                        break
                    nx += dx
                    ny += dy

        return moves

    def getLegalMoves(self, piece):
        # Safety: only generate moves for side to move
        if piece.data[0] != self.sideToMove():
            return set()

        legal = set()
        pseudo = self._getPseudoLegalMoves(piece)

        for move in pseudo:

            src_idx = (piece.data[3] - 1) * 8 + (piece.data[2] - 1)
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
        actions = list()

        # Helper to convert (x,y) -> 0–63 index
        def pos_to_index(pos):
            x, y = pos
            # Chess board: x=1..8 (a..h), y=1..8 (1..8 rank)
            return (y - 1) * 8 + (x - 1)

        for piece in self.board:
            if piece.data[0] == whiteToMove:
                from_idx = pos_to_index(piece.getPosition())
                for to_pos in self.getLegalMoves(piece):
                    to_idx = pos_to_index(to_pos)
                    actions.append((from_idx, to_idx))

        return actions




    def getPiecePositions(self):
        return [(p.data[2],p.data[3],p.data[0]) for p in self.board]

    def pieceAt(self, coord) -> Optional["Piece"]:
        if coord[0] < 1 or coord[1] < 1 or coord[0] > 8 or coord[1] > 8:
            return None
        return self.lookup[coord]

    def isEndgame(self):
        # Endgame if queens are gone OR low material
        queens = sum(1 for p in self.board if p.data[1] == 5)
        major_minor = sum(1 for p in self.board if p.data[1] in (2, 3, 4))
        return queens == 0 or major_minor <= 4

    def pawnAttacksSquare(self, x, y, byWhite):
        """
        Returns True if square (x, y) is attacked by an enemy pawn
        """
        direction = -1 if byWhite else 1
        for dx in (-1, 1):
            px, py = x + dx, y + direction
            if 1 <= px <= 8 and 1 <= py <= 8:
                p = self.pieceAt((px, py))
                if p and p.data[1] == 1 and p.data[0] == byWhite:
                    return True
        return False

    def countPawnIslands(self, isWhite):
        """
        Counts the number of pawn islands for the given color.
        A pawn island is one or more pawns on adjacent files.
        """
        files_with_pawns = [False] * 9  # index 1..8 used

        # Mark files that contain at least one pawn of this color
        for p in self.board:
            if p.data[1] == 1 and p.data[0] == isWhite:
                files_with_pawns[p.data[2]] = True

        islands = 0
        in_island = False

        # Scan files a–h (1–8)
        for file in range(1, 9):
            if files_with_pawns[file]:
                if not in_island:
                    islands += 1
                    in_island = True
            else:
                in_island = False

        return islands

    def backwardPawnPenalty(self, pawn):
        x, y = pawn.data[2], pawn.data[3]
        isWhite = pawn.data[0]
        lookup = pawn.board.lookup if hasattr(pawn, "board") else None

        direction = 1 if isWhite else -1
        front = (x, y + direction)

        # Blocked pawn
        if lookup and front in lookup:
            return 10

        # No friendly pawn support behind on adjacent files
        support = False
        for dx in (-1, 1):
            fx = x + dx
            if 1 <= fx <= 8:
                check_y = y - direction
                p = lookup.get((fx, check_y)) if lookup else None
                if p and p.data[1] == 1 and p.data[0] == isWhite:
                    support = True
                    break

        if not support:
            return 8

        return 0

    def passedPawnValue(self, pawn, furthest):
        """
        Determines if a pawn is "passed", and if so, gives its value. Passed pawns have no enemy pawns that can
        obstruct its movement to the end of the board. The closer a pawn is to the end of the board, the better its
        value.
        :param pawn: The pawn to check
        :param furthest: The row of the furthest pawn of its opposite color
        :return: The pawn's passed pawn value, or 0 if it is not a passed pawn
        """
        direction = 1 if pawn.data[0] else -1

        # Must be ahead of furthest opposing pawn
        # restrict to same or adjacent files
        for fx in range(max(1, pawn.data[2] - 1), min(8, pawn.data[2] + 1) + 1):
            for rank in range(1, 9):
                p = self.pieceAt((fx, rank))
                if p and p.data[1] == 1 and p.data[0] != pawn.data[0]:
                    if pawn.data[0] and rank > pawn.data[3]:
                        return 0
                    if not pawn.data[0] and rank < pawn.data[3]:
                        return 0

        # Check files ahead for enemy pawns
        start_rank = pawn.data[3] + direction
        end_rank = 9 if pawn.data[0] else 0

        for fx in range(max(1, pawn.data[2] - 1), min(8, pawn.data[2] + 1) + 1):
            rank = start_rank
            while rank != end_rank:
                p = self.pieceAt((fx, rank))
                if p and p.data[1] == 1 and p.data[0] != pawn.data[0]:
                    return 0
                rank += direction

        # Advancement-based value
        advancement = pawn.data[3] if pawn.data[0] else 9 - pawn.data[3]
        bonus = int(8 * math.exp(0.58624 * (advancement - 2)))

        # Blocked passed pawn penalty
        if self.pieceAt((pawn.data[2], pawn.data[3] + direction)):
            bonus //= 2

        return bonus

    def connectedPawnValue(self, p):
        direction = 1 if p.data[0] else -1

        connected = False

        # Check adjacent pawns (same rank)
        for dx in (-1, 1):
            neighbor = self.pieceAt((p.data[2] + dx, p.data[3]))
            if neighbor and neighbor.data[1] == 1 and neighbor.data[0] == p.data[0]:
                connected = True

        # Check supporting pawns (one rank ahead)
        for dx in (-1, 1):
            supporter = self.pieceAt((p.data[2] + dx, p.data[3] + direction))
            if supporter and supporter.data[1] == 1 and supporter.data[0] == p.data[0]:
                connected = True

        if not connected:
            return 0

        advancement = p.data[3] if p.data[0] else 9 - p.data[3]
        return int(15 * (0.1 * advancement + 1))

    def isolatedPawnPenalty(self, pawn, pawn_files):
        x = pawn.data[2]

        left = pawn_files.get(x - 1, 0)
        right = pawn_files.get(x + 1, 0)

        if left == 0 and right == 0:
            return 15

        return 0

    def doubledPawnPenalty(self, pawn_count_on_file):
        # No penalty for 0 or 1 pawn
        if pawn_count_on_file <= 1:
            return 0

        # Penalize each extra pawn
        return 12 * (pawn_count_on_file - 1)

    def kingSafetyPenalty(self, king):
        penalty = 0
        x, y = king.data[2], king.data[3]
        lookup = king.board.lookup if hasattr(king, "board") else None

        # Pawn shield penalty
        direction = 1 if king.data[0] else -1
        shield_rank = y + direction

        for dx in (-1, 0, 1):
            file_x = x + dx
            if 1 <= file_x <= 8 and 1 <= shield_rank <= 8:
                p = lookup.get((file_x, shield_rank)) if lookup else None
                if not p or p.data[1] != 1 or p.data[0] != king.data[0]:
                    penalty += 10

        # Open / semi-open files near king
        for dx in (-1, 0, 1):
            file_x = x + dx
            if 1 <= file_x <= 8:
                has_friendly_pawn = False
                for rank in range(1, 9):
                    p = lookup.get((file_x, rank)) if lookup else None
                    if p and p.data[1] == 1 and p.data[0] == king.data[0]:
                        has_friendly_pawn = True
                        break
                if not has_friendly_pawn:
                    penalty += 8

        return penalty

    def getScore(self):
        score = 0
        lookup = self.lookup
        board = self.board

        # -----------------------------
        # Pre-collect pieces by type
        # -----------------------------
        white_pawns = []
        black_pawns = []
        white_rooks = []
        black_rooks = []
        white_king = None
        black_king = None

        for piece in board:
            if piece.data[1] == 1:
                (white_pawns if piece.data[0] else black_pawns).append(piece)
            elif piece.data[1] == 4:
                (white_rooks if piece.data[0] else black_rooks).append(piece)
            elif piece.data[1] == 6:
                if piece.data[0]:
                    white_king = piece
                else:
                    black_king = piece

            # -----------------------------
            # Material (unchanged)
            # -----------------------------
            modifier = 1 if piece.data[0] else -1
            score += piece_values[piece.data[1]][1] * modifier

        # -----------------------------
        # Pawn structure (cached)
        # -----------------------------
        pawn_files_white = {i: 0 for i in range(1, 9)}
        pawn_files_black = {i: 0 for i in range(1, 9)}

        for p in white_pawns:
            pawn_files_white[p.data[2]] += 1
        for p in black_pawns:
            pawn_files_black[p.data[2]] += 1

        # -----------------------------
        # Pawn evaluation
        # -----------------------------
        for pawn in white_pawns:
            modifier = 1
            score += self.passedPawnValue(pawn, pawn_files_black)
            score += self.connectedPawnValue(pawn)
            score -= self.doubledPawnPenalty(pawn_files_white[pawn.data[2]])
            score -= self.isolatedPawnPenalty(pawn, pawn_files_white)
            score -= self.backwardPawnPenalty(pawn)

        for pawn in black_pawns:
            modifier = -1
            score -= self.passedPawnValue(pawn, pawn_files_white)
            score -= self.connectedPawnValue(pawn)
            score += self.doubledPawnPenalty(pawn_files_black[pawn.data[2]])
            score += self.isolatedPawnPenalty(pawn, pawn_files_black)
            score += self.backwardPawnPenalty(pawn)

        # -----------------------------
        # Rook evaluation (open / semi-open files)
        # -----------------------------
        for rook in white_rooks:
            modifier = 1
            if pawn_files_white[rook.data[2]] == 0:
                score += 15
            elif pawn_files_black[rook.data[2]] == 0:
                score += 7

        for rook in black_rooks:
            modifier = -1
            if pawn_files_black[rook.data[2]] == 0:
                score -= 15
            elif pawn_files_white[rook.data[2]] == 0:
                score -= 7

        # -----------------------------
        # King safety (single call each)
        # -----------------------------
        if white_king:
            score -= self.kingSafetyPenalty(white_king)
        if black_king:
            score += self.kingSafetyPenalty(black_king)

        return score

    def makeMove(self, action) -> UndoRecord:
        """
        Makes a move on the board with the given action. This modifies the board state.
        :param action: A tuple (int, int) representing the action to perform
        :return: A record of the board's info before the move, used for undoing moves later
        """
        src = ((action[0] % 8) + 1, (action[0] // 8) + 1)
        dst = ((action[1] % 8) + 1, (action[1] // 8) + 1)

        piece = self.pieceAt(src)
        captured = self.pieceAt(dst)

        undo = UndoRecord(
            moved_piece=piece,
            src=src,
            dst=dst,
            captured_piece=captured,
            captured_square=dst if captured else None,
            old_metadata=bytearray(self.metadata),
            rook_move=None
        )

        # XOR moving piece from old pos
        self.hash ^= ZOBRIST_TABLE[(piece.data[1], int(piece.data[0]), src[0], src[1])]

        # remove captured piece
        if captured:

            # XOR captured piece
            self.hash ^= ZOBRIST_TABLE[(captured.data[1], int(captured.data[0]), dst[0], dst[1])]
            self.board.remove(captured)
            del self.lookup[dst]

        # en passant capture
        if piece.data[1] == 1 and dst == (self.metadata[4], self.metadata[5]):
            cap_y = src[1]
            cap_sq = (dst[0], cap_y)
            ep_piece = self.pieceAt(cap_sq)
            undo.captured_piece = ep_piece
            undo.captured_square = cap_sq

            if ep_piece:
                # XOR out en passant capture
                self.hash ^= ZOBRIST_TABLE[(ep_piece.data[1], int(ep_piece.data[0]), cap_sq[0], cap_sq[1])]
                self.board.remove(ep_piece)
            del self.lookup[cap_sq]

        # move piece
        del self.lookup[src]
        piece.setPosition(*dst)
        self.lookup[dst] = piece

        # XOR moving the piece
        self.hash ^= ZOBRIST_TABLE[(piece.data[1], int(piece.data[0]), dst[0], dst[1])]

        # castling
        if piece.data[1] == 6 and abs(dst[0] - src[0]) == 2:
            if dst[0] == 7:  # kingside
                rook_src = (8, src[1])
                rook_dst = (6, src[1])
            else:  # queenside
                rook_src = (1, src[1])
                rook_dst = (4, src[1])

            rook = self.pieceAt(rook_src)
            if rook:
                # XOR out old rook and XOR in new one
                self.hash ^= ZOBRIST_TABLE[(rook.data[1], int(rook.data[0]), rook_src[0], rook_src[1])]
                del self.lookup[rook_src]
                rook.setPosition(*rook_dst)
                self.lookup[rook_dst] = rook
                self.hash ^= ZOBRIST_TABLE[(rook.data[1], int(rook.data[0]), rook_dst[0], rook_dst[1])]
                undo.rook_move = (rook, rook_src, rook_dst)

        # update metadata
        self.metadata[4] = 0
        self.metadata[5] = 0

        if piece.data[1] == 1 and abs(dst[1] - src[1]) == 2:
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
        self.hash ^= ZOBRIST_TABLE[(piece.data[1], int(piece.data[0]), undo.dst[0], undo.dst[1])]

        if undo.dst in self.lookup:
            del self.lookup[undo.dst]

        piece.setPosition(*undo.src)
        self.lookup[undo.src] = piece

        # XOR the piece back into its source position
        self.hash ^= ZOBRIST_TABLE[(piece.data[1], int(piece.data[0]), undo.src[0], undo.src[1])]

        if undo.captured_piece:
            cap_p = undo.captured_piece
            cap_sq = undo.captured_square
            self.board.add(cap_p)
            self.lookup[cap_sq] = cap_p
            # XOR the captured piece back onto the board
            self.hash ^= ZOBRIST_TABLE[(cap_p.data[1], int(cap_p.data[0]), cap_sq[0], cap_sq[1])]

        if undo.rook_move:
            rook, r_src, r_dst = undo.rook_move
            # XOR out rook from dst, XOR back into src
            self.hash ^= ZOBRIST_TABLE[(rook.data[1], int(rook.data[0]), r_dst[0], r_dst[1])]
            del self.lookup[r_dst]
            rook.setPosition(*r_src)
            self.lookup[r_src] = rook
            self.hash ^= ZOBRIST_TABLE[(rook.data[1], int(rook.data[0]), r_src[0], r_src[1])]

        self.metadata = undo.old_metadata
        self.hash ^= SIDE_HASH

    def kingInCheck(self, white) -> bool:
        """
        Determines if the king is in check.
        :param white: If True, checks for the White king; otherwise, checks for the Black king.
        """
        king = next((p for p in self.board if p.data[1] == 6 and p.data[0] == white), None)
        if not king:
            return False

        kx, ky = king.getPosition()
        enemy_white = not white

        # Check for Knight attacks
        knight_moves = [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]
        for dx, dy in knight_moves:
            p = self.pieceAt((kx + dx, ky + dy))
            if p and p.data[1] == 2 and p.data[0] == enemy_white:
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
                    if p.data[0] == enemy_white and p.data[1] in attackers:
                        return True
                    break  # Blocked by any piece
                tx += dx
                ty += dy

        # 3. Check for Pawn attacks
        pawn_y_dir = 1 if white else -1
        for dx in [-1, 1]:
            p = self.pieceAt((kx + dx, ky + pawn_y_dir))
            if p and p.data[1] == 1 and p.data[0] == enemy_white:
                return True

        # 4. Check for adjacent King (illegal position, but good for safety)
        for dx, dy in itertools.product([-1, 0, 1], repeat=2):
            if dx == 0 and dy == 0: continue
            p = self.pieceAt((kx + dx, ky + dy))
            if p and p.data[1] == 6 and p.data[0] == enemy_white:
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

        sorted_pieces = sorted(self.board, key=lambda p: (p.data[3], p.data[2]))

        for piece in sorted_pieces:
            arr.append(piece.data[2])
            arr.append(piece.data[3])
            arr.append(piece.data[1] if piece.data[0] else 256 - piece.data[1])

        arr.append(0)
        # Add metadata bytes
        arr.extend(self.metadata)

        return bytes(arr)

    def __eq__(self, other):
        return isinstance(other, GameState) and self.hash == other.hash

class Piece:
    """
    The Piece class represents a chess piece, represented with a byte array.
    Byte array is stored as [isWhite, kind(piece type), x, y]

    """
    def __init__(self, isWhite, kind, x=0, y=0):
        self.data = bytearray(b'\0x00\0x00\0x00\0x00')
        self.data[0] = 1 if isWhite else 0
        self.data[1] = kind
        self.data[2] = x
        self.data[3] = y
        
    def __str__(self):
        if self.data[0]:
            color = "White"
        else:
            color = "Black"
        return f"{color} {piece_values[self.data[1]][0]} at {chr(self.data[2]+96)}{self.data[3]}"

    def setPosition(self, x, y):
        self.data[2] = x
        self.data[3] = y

    def getPosition(self):
        return self.data[2], self.data[3]

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
