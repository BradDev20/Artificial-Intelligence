import math
from collections import defaultdict
from typing import Optional

import web_parser
import time, itertools
import numpy as np

#Tuple values are the name, centipawn value, and value per square it can travel to
piece_values = {1: ("Pawn", 100, 0), 2: ("Knight", 325, 5), 3: ("Bishop", 330, 4),
          4: ("Rook", 500, 3), 5: ("Queen", 950, 2), 6: ("King", 0, 0)}
_notation = {'p': 1, 'n': 2, 'b':3, 'r': 4, 'q': 5, 'k': 6}

class GameState:
    def __init__(self, board, captures):
        self.board = []
        self.captures = captures
        self._lookup = defaultdict(lambda: None)

        for piece in board:
            x = int(piece[3])
            y = int(piece[2])
            self.board.append(Piece((piece[0] == 'w'), _notation[piece[1]], x, y))
            self._lookup[(x,y)] = self.board[len(self.board)-1]

        self.board = self.board

    def __str__(self):
        return str([str(piece) for piece in self.board])

    def getAttackSquares(self, isWhite):
        return set(itertools.chain.from_iterable([self.getLegalMoves(p, False) for p in self.board if p.isWhite == isWhite]))

    def pawnAttacksSquare(self, x, y, byWhite):
        """
        Returns True if square (x, y) is attacked by an enemy pawn
        """
        direction = -1 if byWhite else 1
        for dx in (-1, 1):
            px, py = x + dx, y + direction
            if 1 <= px <= 8 and 1 <= py <= 8:
                p = self.pieceAt((px, py))
                if p and p.kind == 1 and p.isWhite == byWhite:
                    return True
        return False

    def getLegalMoves(self, piece, checkCheck=True):
        """
        Gives the legal moves for a piece. Does not consider castling.

        :param piece: A Piece object on the board
        :param checkCheck: An internal checker for if the piece should consider being put into check. Does nothing if the piece is not a king
        :return: a Set of coordinates the piece can move to
        """

        moves = []
        loc = piece.getPosition()


        pos_set = set(self.getPiecePositions())
        whites = {p for p in self.getPiecePositions() if p[2]}
        blacks = {p for p in self.getPiecePositions() if not p[2]}

        def inBounds(position):
            return (0 < position[0] <= 8) and (0 < position[1] <= 8)

        match piece.kind:

            #Pawn
            case 1:

                if piece.isWhite:
                    # Check if the first square is open
                    forward1 = (loc[0], loc[1]+1)
                    if forward1 not in pos_set and inBounds(forward1):
                        moves.append(forward1)

                        # If it is, check if it can move two spaces
                        if piece.y == 2:
                            forward2 = (loc[0], loc[1]+2)
                            if forward2 not in pos_set and inBounds(forward2):
                                moves.append(forward2)

                    # Can the pawn capture on the right?
                    capture_right = (loc[0]+1, loc[1]+1)
                    if capture_right in blacks:
                        moves.append(capture_right)

                    # On the left?
                    capture_left = (loc[0]-1, loc[1]+1)
                    if capture_left in blacks:
                        moves.append(capture_left)

                else:

                    # Check if the first square is open
                    forward1 = (loc[0], loc[1]-1)
                    if forward1 not in pos_set and inBounds(forward1):
                        moves.append(forward1)

                        # If it is, check if it can move two spaces
                        if piece.y == 7:
                            forward2 = (loc[0], loc[1]-2)
                            if forward2 not in pos_set and inBounds(forward2):
                                moves.append(forward2)

                    # Can the pawn capture on the right?
                    capture_right = (loc[0]+1, loc[1]-1)
                    if capture_right in whites:
                        moves.append(capture_right)

                    # On the left?
                    capture_left = (loc[0]-1, loc[1]-1)
                    if capture_left in whites:
                        moves.append(capture_left)

                    return moves

            #Knight, King
            case 2 | 6:

                team = whites if piece.isWhite else blacks
                targets = []

                #Knight's 8 possible jump locations
                if piece.kind == 2:
                    targets = [(-1, 2, False), (1, 2, False), (2, 1, False), (2, -1, False),
                                (1, -2, False), (-1, -2, False), (-2, 1, False), (-2, -1, False)]

                    for t in targets:
                        target_loc = (loc[0]+t[0], loc[1]+t[1])
                        if target_loc not in team and inBounds(target_loc):
                            moves.append(target_loc)  # Capture opponent

                #King's 8 possible moves
                else:
                    targets = [(1, 0, False), (-1, 0, False), (1, 1, False), (1, -1, False),
                               (-1, 1, False), (-1, -1, False), (0, 1, False), (0, -1, False)]
                    dangerous = {}
                    if checkCheck:
                        dangerous = set(itertools.chain.from_iterable([self.getLegalMoves(p, False) for p in self.board if p.isWhite != piece.isWhite]))

                    for t in targets:
                        target_loc = (loc[0]+t[0], loc[1]+t[1])
                        if target_loc not in team and inBounds(target_loc):
                            # If an enemy piece can move there, don't make it an option
                            # Doesn't account for the king getting into check by capturing a piece but it will be checked elsewhere
                            if checkCheck:
                                if target_loc not in dangerous:
                                    moves.append(target_loc)
                            else:
                                moves.append(target_loc)



            #Bishop, Rook, Queen
            case 3 | 4 | 5:
                directions = []
                #Bishop's options
                if piece.kind == 3:
                    directions = [(1, 1, False), (1, -1, False), (-1, 1, False), (-1, -1, False)]  # NE, SE, NW, SW

                #Rook's options
                if piece.kind == 4:
                    directions = [(1, 0, False), (-1, 0, False), (0, 1, False), (0, -1, False)] # N, S, E, W

                #Queen's options
                if piece.kind == 5:
                    directions = [(1, 1, False), (1, -1, False), (-1, 1, False), (-1, -1, False),
                                  (1, 0, False), (-1, 0, False), (0, 1, False), (0, -1, False)]  # Compass rose

                team = whites if piece.isWhite else blacks
                for d in directions:
                    ndir = (loc[0]+d[0], loc[1]+d[1])
                    while 0 < ndir[0] <= 8 and 0 < ndir[1] <= 8:
                        if ndir in pos_set:
                            if ndir in team:
                                break  # Blocked by own piece
                            else:
                                moves.append(ndir)  # Capture opponent
                                break
                        moves.append(ndir)  # Empty square
                        ndir = (ndir[0]+d[0], ndir[1]+d[1])

        return set([(x,y) for x,y in moves])


    def getPiecePositions(self):
        return [(p.x,p.y,p.isWhite) for p in self.board]

    def pieceAt(self, coord) -> Optional["Piece"]:
        if coord[0] < 1 or coord[1] < 1 or coord[0] > 8 or coord[1] > 8:
            return None
        return self._lookup[coord]

    def isEndgame(self):
        # Endgame if queens are gone OR low material
        queens = sum(1 for p in self.board if p.kind == 5)
        major_minor = sum(1 for p in self.board if p.kind in (2, 3, 4))
        return queens == 0 or major_minor <= 4

    def getScore(self):
        """
        Determines the score for the chessboard, in centipawns.
        A positive score means White has an advantage, and a negative score means Black has an advantage.
        :return: The score, which is an integer
        """

        legalMovesBlack = set(itertools.chain.from_iterable([self.getLegalMoves(p) for p in self.board if not p.isWhite]))
        legalMovesWhite = set(itertools.chain.from_iterable([self.getLegalMoves(p) for p in self.board if p.isWhite]))

        def pawnAttacksSquare(x, y, byWhite):
            """
            Returns True if square (x, y) is attacked by an enemy pawn
            """
            direction = -1 if byWhite else 1
            for dx in (-1, 1):
                px, py = x + dx, y + direction
                if 1 <= px <= 8 and 1 <= py <= 8:
                    p = self.pieceAt((px, py))
                    if p and p.kind == 1 and p.isWhite == byWhite:
                        return True
            return False

        def countPawnIslands(isWhite):
            """
            Counts the number of pawn islands for the given color.
            A pawn island is one or more pawns on adjacent files.
            """
            files_with_pawns = [False] * 9  # index 1..8 used

            # Mark files that contain at least one pawn of this color
            for p in self.board:
                if p.kind == 1 and p.isWhite == isWhite:
                    files_with_pawns[p.x] = True

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

        def isBackwardPawn(pawn, far_back):
            """
            A pawn is backward if:
            1) It is the furthest-back pawn on its file group
            2) It has no friendly pawn on adjacent files behind or level with it
            3) The square in front is controlled by an enemy pawn
            """

            if pawn.kind != 1:
                return False

            # Must be the furthest-back pawn
            if pawn.y != far_back:
                return False

            direction = 1 if pawn.isWhite else -1
            forward_y = pawn.y + direction

            if not (1 <= forward_y <= 8):
                return False

            # ----------------------------------
            # Check for adjacent pawn support
            # ----------------------------------
            for dx in (-1, 1):
                fx = pawn.x + dx
                if not (1 <= fx <= 8):
                    continue

                for rank in range(1, pawn.y + 1) if pawn.isWhite else range(pawn.y, 9):
                    p = self.pieceAt((fx, rank))
                    if p and p.kind == 1 and p.isWhite == pawn.isWhite:
                        return False

            # ----------------------------------
            # Check if advance square is unsafe
            # ----------------------------------
            if self.pieceAt((pawn.x, forward_y)):
                return False

            enemy_is_white = not pawn.isWhite
            if self.pawnAttacksSquare(pawn.x, forward_y, enemy_is_white):
                return True

            return False

        def passedPawnValue(pawn, furthest):
            """
            Determines if a pawn is "passed", and if so, gives its value. Passed pawns have no enemy pawns that can
            obstruct its movement to the end of the board. The closer a pawn is to the end of the board, the better its
            value.
            :param pawn: The pawn to check
            :param furthest: The row of the furthest pawn of its opposite color
            :return: The pawn's passed pawn value, or 0 if it is not a passed pawn
            """
            direction = 1 if pawn.isWhite else -1

            # Must be ahead of furthest opposing pawn
            # restrict to same or adjacent files
            for fx in range(max(1, pawn.x - 1), min(8, pawn.x + 1) + 1):
                for rank in range(1, 9):
                    p = self.pieceAt((fx, rank))
                    if p and p.kind == 1 and p.isWhite != pawn.isWhite:
                        if pawn.isWhite and rank > pawn.y:
                            return 0
                        if not pawn.isWhite and rank < pawn.y:
                            return 0

            # Check files ahead for enemy pawns
            start_rank = pawn.y + direction
            end_rank = 9 if pawn.isWhite else 0

            for fx in range(max(1, pawn.x - 1), min(8, pawn.x + 1) + 1):
                rank = start_rank
                while rank != end_rank:
                    p = self.pieceAt((fx, rank))
                    if p and p.kind == 1 and p.isWhite != pawn.isWhite:
                        return 0
                    rank += direction

            # Advancement-based value
            advancement = pawn.y if pawn.isWhite else 9 - pawn.y
            bonus = int(8 * math.exp(0.58624 * (advancement - 2)))

            # Blocked passed pawn penalty
            if self.pieceAt((pawn.x, pawn.y + direction)):
                bonus //= 2

            return bonus

        def connectedPawnValue(p):
            direction = 1 if p.isWhite else -1

            connected = False

            # Check adjacent pawns (same rank)
            for dx in (-1, 1):
                neighbor = self.pieceAt((p.x + dx, p.y))
                if neighbor and neighbor.kind == 1 and neighbor.isWhite == p.isWhite:
                    connected = True

            # Check supporting pawns (one rank ahead)
            for dx in (-1, 1):
                supporter = self.pieceAt((p.x + dx, p.y + direction))
                if supporter and supporter.kind == 1 and supporter.isWhite == p.isWhite:
                    connected = True

            if not connected:
                return 0

            advancement = p.y if p.isWhite else 9 - p.y
            return int(15 * (0.1 * advancement + 1))

        def isDoubledPawn(p):
            return sum(
                1 for q in self.board
                if q.kind == 1 and q.isWhite == p.isWhite and q.x == p.x
            ) > 1

        """
        -----------------------------
        *****************************
        START OF MAIN EVALUATION CODE
        *****************************
        -----------------------------
        """

        score = 0
        bishop_count_white = 0
        bishop_count_black = 0
        #The center four squares
        center = {(4,4), (4,5), (5,4), (5,5)}

        ISLAND_PENALTY = 10

        white_islands = countPawnIslands(True)
        black_islands = countPawnIslands(False)

        # First island is free
        if white_islands > 1:
            score -= (white_islands - 1) * ISLAND_PENALTY

        if black_islands > 1:
            score += (black_islands - 1) * ISLAND_PENALTY


        # Check if rooks are connected
        rooks_white = [p for p in self.board if p.kind == 4 and p.isWhite]
        rooks_black = [p for p in self.board if p.kind == 4 and not p.isWhite]
        if len(rooks_white) == 2:
            if rooks_white[0].x == rooks_white[1].x:
                for _ in range(min(rooks_white[0].y, rooks_white[1].y)+1, max(rooks_white[0].y, rooks_white[1].y)):
                    if self.pieceAt((rooks_white[0].x, _)):
                        break
                else:
                    score += 15

            elif rooks_white[0].y == rooks_white[1].y:
                for _ in range(min(rooks_white[0].x, rooks_white[1].x)+1, max(rooks_white[0].x, rooks_white[1].x)):
                    if self.pieceAt((_, rooks_white[0].y)):
                        break
                else:
                    score += 15

        if len(rooks_black) == 2:
            if rooks_black[0].x == rooks_black[1].x:
                for _ in range(min(rooks_black[0].y, rooks_black[1].y)+1, max(rooks_black[0].y, rooks_black[1].y)):
                    if self.pieceAt((rooks_black[0].x, _)):
                        break
                else:
                    score -= 15

            elif rooks_black[0].y == rooks_black[1].y:
                for _ in range(min(rooks_black[0].x, rooks_black[1].x)+1, max(rooks_black[0].x, rooks_black[1].x)):
                    if self.pieceAt((_, rooks_black[0].y)):
                        break
                else:
                    score -= 15


        """
        ------------------
        ******************
        PER-PIECE FOR LOOP
        ******************
        ------------------
        """
        #For each piece, add its face value, plus the value of each square it can move to
        for piece in self.board:

            modifier = 1 if piece.isWhite else -1
            score += piece_values[piece.kind][1] * modifier
            moves = self.getLegalMoves(piece)
            center_hits = 0
            for m in moves:
                if m in center and piece.kind != 6:
                    center_hits += 1
                else:
                    score += piece_values[piece.kind][2] * modifier

            score += min(center_hits, 2) * 7 * modifier

            match piece.kind:

                case 1:

                    if piece.isWhite:
                        far_back = min(p.y for p in self.board if p.kind == 1 and p.isWhite)
                        far_pass = max((p.y for p in self.board if p.kind == 1 and not p.isWhite), default=0)
                    else:
                        far_back = max(p.y for p in self.board if p.kind == 1 and not p.isWhite)
                        far_pass = min((p.y for p in self.board if p.kind == 1 and p.isWhite), default=9)

                    if isBackwardPawn(piece, far_back):
                        score -= 12 * modifier

                    score += passedPawnValue(piece, far_pass) * modifier

                    #subtract score if the pawn can't move forwards
                    direction = 1 if piece.isWhite else -1
                    if self.pieceAt((piece.x, piece.y + direction)):
                        score -= 9 * modifier

                    #subtract score if a pawn is "isolated"(no friendly pawns in adjacent files)
                    isolated = True
                    for i in range(1, 9):
                        if i not in (piece.x - 1, piece.x + 1):
                            continue
                        for rank in range(1, 9):
                            adj_pawn = self.pieceAt((i, rank))
                            if adj_pawn and adj_pawn.kind == 1 and adj_pawn.isWhite == piece.isWhite:
                                isolated = False
                                break
                        if not isolated:
                            break

                    score -= (10 * modifier) if isolated else 0

                    #add score if a pawn is connected to another pawn on its right/left
                    #since this goes through every pawn it will handle longer chains
                    #score increases the further the pawn is down the board
                    score += connectedPawnValue(piece) * modifier

                    #subtract score if a pawn is "doubled"(there is another friendly pawn in its file or column)
                    score -= (20 * modifier) if isDoubledPawn(piece) else 0

                case 2:  # Knight

                    # Rim penalty
                    if piece.x == 1 or piece.x == 8 or piece.y == 1 or piece.y == 8:
                        score -= (20 * modifier)

                    # Safe from enemy pawns
                    if not pawnAttacksSquare(piece.x, piece.y, not piece.isWhite):
                        score += 20 * modifier

                        # Pawn support bonus
                        direction = -1 if piece.isWhite else 1
                        for dx in (-1, 1):
                            p = self.pieceAt((piece.x + dx, piece.y + direction))
                            if p and p.kind == 1 and p.isWhite == piece.isWhite:
                                score += 10 * modifier
                                break
                case 3:

                    if piece.isWhite:
                        bishop_count_white += 1
                    else:
                        bishop_count_black += 1

                    moves = self.getLegalMoves(piece)
                    if any(c in moves for c in center):
                        for _ in {(p.x,p.y) for p in self.board if p.kind == 1 and p.isWhite == piece.isWhite}:
                            if _ in moves:
                                score -= (18 * modifier)

                case 4:  # Rook

                    # ----------------------------------
                    # OPEN / SEMI-OPEN FILE
                    # ----------------------------------
                    own_pawn = False
                    enemy_pawn = False

                    for rank in range(1, 9):
                        p = self.pieceAt((piece.x, rank))
                        if p and p.kind == 1:
                            if p.isWhite == piece.isWhite:
                                own_pawn = True
                            else:
                                enemy_pawn = True

                    # Open file: no pawns at all
                    if not own_pawn and not enemy_pawn:
                        score += 20 * modifier

                    # Semi-open file: enemy pawn only
                    elif not own_pawn and enemy_pawn:
                        score += 10 * modifier

                    # ----------------------------------
                    # 7th / 2nd RANK ACTIVITY
                    # ----------------------------------
                    if piece.isWhite:
                        if piece.y == 7:
                            # Enemy pawns or king on 7th
                            if any(
                                    (p.kind == 1 or p.kind == 6) and not p.isWhite
                                    for p in self.board
                                    if p.y == 7
                            ):
                                score += 25 * modifier
                    else:
                        if piece.y == 2:
                            if any(
                                    (p.kind == 1 or p.kind == 6) and p.isWhite
                                    for p in self.board
                                    if p.y == 2
                            ):
                                score += 25 * modifier

                    # ----------------------------------
                    # ROOK ON OPEN / SEMI-OPEN 7th
                    # ----------------------------------
                    if (piece.isWhite and piece.y == 7) or (not piece.isWhite and piece.y == 2):
                        if not own_pawn:
                            score += 10 * modifier

                case 5:  # Queen

                    # -------------------------
                    # Early queen development
                    # -------------------------
                    # Penalize if queen leaves back rank while minor pieces are undeveloped

                    starting_rank = 1 if piece.isWhite else 8
                    if piece.y != starting_rank:

                        undeveloped_minors = 0

                        for p in self.board:
                            if p.isWhite == piece.isWhite and p.kind in (2, 3):  # Knight or Bishop
                                if p.y == starting_rank:
                                    undeveloped_minors += 1

                        # −5 per undeveloped minor, capped at −30
                        early_penalty = min(5 * undeveloped_minors, 30)
                        score -= early_penalty * modifier

                    # -------------------------
                    # Queen proximity to own king
                    # -------------------------
                    # Discourage awkward queen placements near own king

                    king = next(p for p in self.board if p.kind == 6 and p.isWhite == piece.isWhite)

                    dx = abs(piece.x - king.x)
                    dy = abs(piece.y - king.y)
                    chebyshev_distance = max(dx, dy)

                    if chebyshev_distance <= 2:
                        score -= 10 * modifier

                case 6:  # King

                    if not self.isEndgame():
                        # -------------------------
                        # MIDGAME KING SAFETY
                        # -------------------------

                        # 1. Castling / back-rank penalty
                        starting_rank = 1 if piece.isWhite else 8

                        # Determine if king is castled by position
                        castled = (
                                (piece.x == 7 and piece.y == starting_rank) or  # king-side
                                (piece.x == 3 and piece.y == starting_rank)  # queen-side
                        )

                        # King on back rank but not castled → penalty
                        if piece.y == starting_rank and not castled:

                            # Optional: confirm rook structure (safer)
                            rook_positions = [(1, starting_rank), (8, starting_rank)]
                            has_rook = any(
                                (self.pieceAt(pos) and
                                 self.pieceAt(pos).kind == 4 and
                                 self.pieceAt(pos).isWhite == piece.isWhite)
                                for pos in rook_positions
                            )

                            # Penalize more if rooks still exist
                            if has_rook:
                                score -= 30 * modifier
                            else:
                                score -= 15 * modifier

                        # 2. Pawn shield (f, g, h pawns)
                        shield_files = [piece.x - 1, piece.x, piece.x + 1]
                        shield_penalty = 0

                        for fx in shield_files:
                            if 1 <= fx <= 8:
                                pawn = self.pieceAt((fx, piece.y + (1 if piece.isWhite else -1)))
                                if not pawn or pawn.kind != 1 or pawn.isWhite != piece.isWhite:
                                    shield_penalty += 10

                        score -= shield_penalty * modifier

                        # 3. Center exposure
                        if piece.x in (4, 5) and piece.y in (4, 5):
                            score -= 20 * modifier

                    else:
                        # -------------------------
                        # ENDGAME KING ACTIVITY
                        # -------------------------

                        # Encourage centralization
                        center_x, center_y = 4.5, 4.5
                        dist = abs(piece.x - center_x) + abs(piece.y - center_y)

                        # Max bonus ≈ +30 when very central
                        score += int((6 - dist) * 5) * modifier

        return score


class Piece:
    """
    The Piece class represents a chess piece. It has an x and y coordinate representing its place on the board,
    as a color represented by the boolean isWhite.

    Two pieces are considered "equal" if they are of the same type and color (White rook == White rook even when they are not in the same place).

    While you can declare a piece without giving a location, you should only do so for piece equality comparison.

    """
    def __init__(self, isWhite, kind, x=0, y=0):
        self.kind = kind
        self.x = x
        self.y = y
        self.isWhite = isWhite

    def __str__(self):
        if self.isWhite:
            color = "White"
        else:
            color = "Black"
        return f"{color} {piece_values[self.kind][0]} at ({self.x},{self.y})"

    def __eq__(self, other):
        if not other:
            return False
        return self.kind == other.kind and self.isWhite == other.isWhite

    def setPosition(self, x, y):
        self.x = x
        self.y = y

    def getPosition(self):
        return self.x, self.y

def main():

    web_parser.open_site()

    prev = None

    while True:
        try:
            time.sleep(1.5)
            pieces, captures = web_parser.get_board_data()
            state = GameState(pieces, captures)

            print(state.getScore())
        except Exception as e:
            print("ERROR:",e)
            exit(1)

if __name__ == "__main__":
    main()
