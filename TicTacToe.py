import copy
from typing import Tuple

class GameState:
    def __init__(self, whoseTurn="x", board = None):
        self.whoseTurn = whoseTurn
        if board:
            self.board = board
        else:
            self.board = Board()


class Board:
    def __init__(self, tiles=None):
        if tiles:
            self.tiles = tiles
        else:
            self.tiles = [["*" for _ in range(0, 3)] for _ in range(0, 3)]

    def __repr__(self):
        ans = ""
        for row in self.tiles:
            for col in row:
                ans += str(col)
            ans += "\n"
        return ans
    

class TicTacToe:

    def get_legal_moves(state: GameState):
        board = state.board

        ans = []

        numRows = len(board.tiles)
        numCols = len(board.tiles[0])
        for row in range(0, numRows):
            for col in range(0, numCols):
                if board.tiles[row][col] == "*":
                    move = (row, col)
                    ans.append(move)
        return ans
    
    def apply_move(state: GameState, move: Tuple[int]):
        whoseTurn = state.whoseTurn
        board = state.board
        row, col = move

        newBoard = copy.deepcopy(board)

        newBoard.tiles[row][col] = whoseTurn
        if whoseTurn == "x":
            nextTurn = "o"
        elif whoseTurn == "o":
            nextTurn = "x"
        return GameState(nextTurn, newBoard)

    def is_over(state: GameState) -> [bool, str | None]:
        # returns True/False about whether game is over,
        # and if the game has a winner, it returns the winner

        board = state.board.tiles
        winning_triplets = [
            ((0, 0), (0, 1), (0, 2)),
            ((1, 0), (1, 1), (1, 2)),
            ((2, 0), (2, 1), (2, 2)),
            
            ((0, 0), (1, 0), (2, 0)),
            ((0, 1), (1, 1), (2, 1)),
            ((0, 2), (1, 2), (2, 2)),
            
            ((0, 0), (1, 1), (2, 2)),
            ((2, 0), (1, 1), (0, 2)),
        ]

        for triplet in winning_triplets:
            coord1, coord2, coord3 = triplet
            
            row1, col1 = coord1
            row2, col2 = coord2
            row3, col3 = coord3
            
            letter1 = board[row1][col1]
            letter2 = board[row2][col2]
            letter3 = board[row3][col3]


            if [letter1, letter2, letter3] == ["x", "x", "x"]:
                return True, "x"
            elif [letter1, letter2, letter3] == ["o", "o", "o"]:
                return True, "o"
        
        for row in board:
            for entry in row:
                if entry == "*":
                    return False, None


        return True, None
