import chess
import chess.svg
import chess.pgn
import chess.engine
import chess.polyglot as cpg
import traceback
import webbrowser
import time
import pyttsx3
import numpy as np
from uuid import uuid1
from collections import defaultdict
from flask import Flask, Response, request

class ChessGame():
    def __init__(self, open_game=8) -> None:
        self.board = chess.Board()

        self.move_counter = 0
        self.open_game = open_game
        
        self.white = chess.WHITE
        self.black = chess.BLACK
        self.pawn = chess.PAWN
        self.knight = chess.KNIGHT
        self.bishop = chess.BISHOP
        self.rook = chess.ROOK
        self.queen = chess.QUEEN
        self.king = chess.KING
        self.piece_values = [100, 320, 330, 500, 900, 20000]
        self.pieces = [self.pawn, self.knight, self.bishop, self.rook, self.queen, self.king]
        
        # table for piece square tables
        pawn_table = [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, -20, -20, 10, 10, 5,
            5, -5, -10, 0, 0, -10, -5, 5,
            0, 0, 0, 20, 20, 0, 0, 0,
            5, 5, 10, 25, 25, 10, 5, 5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
            0, 0, 0, 0, 0, 0, 0, 0]
        knight_table = [
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -30, 5, 10, 15, 15, 10, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50]
        bishop_table = [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 5, 0, 0, 0, 0, 5, -10,
            -10, 10, 10, 10, 10, 10, 10, -10,
            -10, 0, 10, 10, 10, 10, 0, -10,
            -10, 5, 5, 10, 10, 5, 5, -10,
            -10, 0, 5, 10, 10, 5, 0, -10,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -20, -10, -10, -10, -10, -10, -10, -20]
        rook_table = [
            0, 0, 0, 5, 5, 0, 0, 0,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            5, 10, 10, 10, 10, 10, 10, 5,
            0, 0, 0, 0, 0, 0, 0, 0]
        queen_table = [
            -20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 5, 5, 5, 5, 5, 0, -10,
            0, 0, 5, 5, 5, 5, 0, -5,
            -5, 0, 5, 5, 5, 5, 0, -5,
            -10, 0, 5, 5, 5, 5, 0, -10,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -20, -10, -10, -5, -5, -10, -10, -20]
        king_table = [
            20, 30, 10, 0, 0, 10, 30, 20,
            20, 20, 0, 0, 0, 0, 20, 20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            -20, -30, -30, -40, -40, -30, -30, -20,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30]
        
        self.piece_square_tables = [pawn_table, knight_table, bishop_table, rook_table, queen_table, king_table]
        
    def get_board(self):
        return self.board

    def get_move_counter(self):
        return self.move_counter

    def get_pieces(self):
        return self.pieces

    def get_piece_square_tables(self):
        return self.piece_square_tables

    def increment_move_counter(self):
        self.move_counter += 1

    def is_opening(self):
        return self.move_counter < self.open_game

    def num_pieces(self, color):
        num_pawns = len(self.board.pieces(self.pawn, color))
        num_knights = len(self.board.pieces(self.knight, color))
        num_bishops = len(self.board.pieces(self.bishop, color))
        num_rooks = len(self.board.pieces(self.rook, color))
        num_queens = len(self.board.pieces(self.queen, color))
        num_king = len(self.board.pieces(self.king, color))
        
        return [num_pawns, num_knights, num_bishops, num_rooks, num_queens, num_king]

    def get_material(self, num_white_pieces, num_black_pieces):
        n = len(self.piece_values)
        
        material_score = 0
        for i in range(n):
            material_score += self.piece_values[i] * (num_white_pieces[i] - num_black_pieces[i] )
        
        return material_score

    def get_squares(self, table, piece):
        squares = sum([table[i] for i in self.board.pieces(piece, self.white)])
        squares += sum([-table[chess.square_mirror(i)] for i in self.board.pieces(piece, self.black)])
        
        return squares

class ChessAI():
    def __init__(self, game: ChessGame, max_depth: int, max_quiescence_depth: int):
        self.game = game
        
        self.max_depth = max_depth
        self.max_quiescence_depth = max_quiescence_depth

        self.transposition_table = TranspositionTable()

    def get_move(self) -> chess.Move:
        # get the current board
        board = self.game.get_board()

        self.game.increment_move_counter()

        # get the opening book move
        opening_book_move = self.get_opening_book_move(board)
        # check if the opening book move is not None
        if opening_book_move is not None:
                return opening_book_move

        # get the best move
        alpha = -np.inf
        beta = np.inf
        depth = 0

        [best_move, _] = self.negamax_alpha_beta(board, alpha, beta, depth)

        # print("best move: ", best_move)
        # return the best move
        return best_move

    def get_opening_book_move(self, board):
        reader = cpg.MemoryMappedReader("/Users/tylertrogden/Documents/CS470/code/proj/pecg_book.bin")
        move = reader.get(board)
        reader.close()

        if move is not None:
            return move.move
        
        return None

    def negamax_alpha_beta(self, board, alpha, beta, depth):
        best_move = chess.Move.null()
        max_value = -np.inf

        tt_entry = 
        
        if depth > self.max_depth:
            print("depth: ", depth, " move: ", best_move, " alpha: ", alpha, " beta: ", beta)
            return [best_move, self.quiescence_search(board, alpha, beta, 0)]
        
        for move in board.legal_moves:
            print("depth: ", depth, " move: ", move, " alpha: ", alpha, " beta: ", beta)
            board.push(move)
            board_value = -self.negamax_alpha_beta(board, -beta, -alpha, depth + 1)[-1]
            board.pop()
            
            if board_value >= beta:
                return [best_move, board_value] # fail soft beta-cutoff
                
            if board_value > max_value:
                max_value = board_value
                best_move = move

                if board_value > alpha:
                    alpha = board_value
            
        return [best_move, max_value]

    def quiescence_search(self, board, alpha, beta, depth):
        stand_pat_score = self.evaluation(board)

        if depth > self.max_quiescence_depth:
            return stand_pat_score
        
        if stand_pat_score >= beta:
            return beta
        
        if alpha < stand_pat_score:
            alpha = stand_pat_score
            
        for move in board.legal_moves:
            if board.is_capture(move):
                board.push(move)
                score = -self.quiescence_search(board, -beta, -alpha, depth + 1)
                board.pop()
                
                if( score >= beta ):
                    return beta
                
                if( score > alpha ):
                    alpha = score

        return alpha

    def evaluation(self, board):
        # get the piece values
        pieces = self.game.get_pieces()

        # get the piece square tables
        tables = self.game.get_piece_square_tables()

        if board.is_checkmate():
            if board.turn:
                return -9999
            else:
                return 9999
        
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0

        else:
            num_white_pieces = self.game.num_pieces(chess.WHITE)
            num_black_pieces = self.game.num_pieces(chess.BLACK)

            material = self.game.get_material(num_white_pieces, num_black_pieces)

            squares = 0
            for i in range(len(tables)):
                squares += self.game.get_squares(tables[i], pieces[i])

            score = material + squares

            if board.turn:
                return score
            else:
                return -score

class TranspositionTable():
    def __init__(self):
        self.size = (64, 12)
        self.table = np.zeros(self.size, dtype=np.int64)
        self.transposition_table = defaultdict(TTEntry)

        self.piece_constants = {
            (chess.PAWN, chess.WHITE): 1, 
            (chess.KNIGHT, chess.WHITE): 2, 
            (chess.BISHOP, chess.WHITE): 3, 
            (chess.ROOK, chess.WHITE): 4, 
            (chess.QUEEN, chess.WHITE): 5, 
            (chess.KING, chess.WHITE): 6, 
            (chess.PAWN, chess.BLACK): 7, 
            (chess.KNIGHT, chess.BLACK): 8, 
            (chess.BISHOP, chess.BLACK): 9, 
            (chess.ROOK, chess.BLACK): 10, 
            (chess.QUEEN, chess.BLACK): 11, 
            (chess.KING, chess.BLACK): 12
         }

        for i in range(64):
            for j in range(12):
                self.table[i, j] = int.from_bytes(uuid1().bytes, byteorder='big', signed=True) >> 64

    def get(self, board):
        key = self.zobrist_key(board)
        if key in self.transposition_table:
            return self.transposition_table[key]

        return None

    def put(self, board, depth) -> None:
        key = self.zobrist_key(board)
        entry = TTEntry(key, depth, board, "EXACT")
        self.transposition_table[key] = entry

    def zobrist_key(self, board: chess.Board):
        key = 0
        if not board.turn:
            key ^= int.from_bytes(uuid1().bytes, byteorder='big', signed=True) >> 64

        for i in range(64):
            piece = board.piece_at(i)
            if piece is not None:
                j = self.piece_constants[(piece.piece_type, piece.color)]
                key ^= self.table[i, j]

        return key
        
class TTEntry():
    def __init__(self, key: int, depth: int, score: float, best_move: chess.Move, flag: str):
        self.key = key
        self.depth = depth
        self.score = score
        self.best_move = best_move
        self.flag = flag

    def __getattribute__(self, __name: str):
        match __name:
            case "key":
                return  self.key
            case "depth":
                return self.depth
            case "score":
                return self.score
            case "best_move":
                return self.best_move
            case "flag":
                return self.flag
            case _:
                return None


app = Flask(__name__)

# Searching Ai's Move
def aimove():
    move = ai.get_move()
    # speak("I am moving " + str(move))
    board.push(move)

# Searching Stockfish's Move
def stockfish():
    engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
    move = engine.play(board, chess.engine.Limit(time=0.1))
    # speak("I am moving " + str(move.move))
    game.board.push(move.move)    

# Speak Function for the Assistant to speak
# def speak(text):
#     engine = pyttsx3.init('nsss')

#     voices = engine.getProperty('voices')
#     engine.setProperty('voice', voices[1].id)  # female voice
#     engine.setProperty('rate', 150)  # speed percent (can go over 100)

#     engine.say(text)

#     engine.runAndWait()
#     engine.stop()

app = Flask(__name__)

# Front Page of the Flask Web Page
@app.route("/")
def main():
    global ai, board, count
    ret = '<html><head>'
    ret += '<style>input {font-size: 20px; } button { font-size: 20px; }</style>'
    ret += '</head><body>'
    ret += '<img width=510 height=510 src="/board.svg?%f"></img></br></br>' % time.time()
    ret += '<form action="/game/" method="post"><button name="New Game" type="submit">New Game</button></form>'
    ret += '<form action="/undo/" method="post"><button name="Undo" type="submit">Undo Last Move</button></form>'
    ret += '<form action="/move/"><input type="submit" value="Make Human Move:"><input name="move" type="text"></input></form>'
    ret += '<form action="/dev/" method="post"><button name="Comp Move" type="submit">Make AI Move</button></form>'
    ret += '<form action="/engine/" method="post"><button name="Stockfish Move" type="submit">Make Stockfish Move</button></form>'

    # if board.is_stalemate():
    #     speak("Draw by stalemate")
    # elif board.is_checkmate():
    #     speak("Checkmate")
    # elif board.is_insufficient_material():
    #     speak("Draw due to insufficient material")
    # elif board.is_check():
    #     speak("Check")

    return ret

# Display Board
@app.route("/board.svg/")
def board():
    return Response(chess.svg.board(board=board, size=700), mimetype='image/svg+xml')

# Human Move
@app.route("/move/")
def move():
    try:
        move = request.args.get('move', default="")
        board.push_san(move)
    except Exception:
        traceback.print_exc()
    
    return main()

# Make AI’s Move
@app.route("/dev/", methods=['POST'])
def dev():
    try:
        aimove()
    except Exception:
        traceback.print_exc()
    
    return main()

# Make UCI Compatible engine's move
@app.route("/engine/", methods=['POST'])
def engine():
    try:
        stockfish()
    except Exception:
        traceback.print_exc()
    
    return main()

# New Game
@app.route("/game/", methods=['POST'])
def game():
    board.reset()
    
    return main()

# Undo
@app.route("/undo/", methods=['POST'])
def undo():
    try:
        board.pop()
    except Exception:
        traceback.print_exc()
        
    return main()

if __name__ == "__main__":
    # Run Flask Web Page and begin game
    game = ChessGame()
    ai = ChessAI(game, 3, 3)
    board = game.get_board()
    count = game.get_move_counter()

    webbrowser.open("http://127.0.0.1:5000/")
    app.run()