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

# ML imports
import os
import torch
import numpy as np
import base64
import pytorch_lightning as pl
import torch.nn.functional as F
from peewee import *
from torch import nn
from torch.utils.data import Dataset, DataLoader, IterableDataset, random_split
from random import randrange
from collections import OrderedDict

db = SqliteDatabase('../2021-07-31-lichess-evaluations-37MM.db')

class Evaluations(Model):
  id = IntegerField()
  fen = TextField()
  binary = BlobField()
  eval = FloatField()

  class Meta:
    database = db

  def binary_base64(self):
    return base64.b64encode(self.binary)

db.connect()

class EvaluationDataset(IterableDataset):
  def __init__(self, count):
    self.count = count
  def __iter__(self):
    return self
  def __next__(self):
    idx = randrange(self.count)
    return self[idx]
  def __len__(self):
    return self.count
  def __getitem__(self, idx):
    eval = Evaluations.get(Evaluations.id == idx+1)
    bin = np.frombuffer(eval.binary, dtype=np.uint8)
    bin = np.unpackbits(bin, axis=0).astype(np.single) 
    eval.eval = max(eval.eval, -15)
    eval.eval = min(eval.eval, 15)
    ev = np.array([eval.eval]).astype(np.single) 
    return {'binary':bin, 'eval':ev}    

LABEL_COUNT = 37164639
dataset = EvaluationDataset(count=LABEL_COUNT)

class EvaluationModel(pl.LightningModule):
  def __init__(self,learning_rate=1e-3,batch_size=1024,layer_count=10):
    super().__init__()
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    layers = []
    for i in range(layer_count-1):
      layers.append((f"linear-{i}", nn.Linear(808, 808)))
      layers.append((f"relu-{i}", nn.ReLU()))
    layers.append((f"linear-{layer_count-1}", nn.Linear(808, 1)))
    self.seq = nn.Sequential(OrderedDict(layers))

  def forward(self, x):
    return self.seq(x)

  def training_step(self, batch, batch_idx):
    x, y = batch['binary'], batch['eval']
    y_hat = self(x)
    loss = F.l1_loss(y_hat, y)
    self.log("train_loss", loss)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

  def train_dataloader(self):
    dataset = EvaluationDataset(count=LABEL_COUNT)
    return DataLoader(dataset, batch_size=self.batch_size, num_workers=2, pin_memory=True)

model_4_layer = EvaluationModel(layer_count=4, batch_size=512, learning_rate=1e-3)
model_6_layer = EvaluationModel(layer_count=6, batch_size=1024, learning_rate=1e-3)

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
    def __init__(self, game: ChessGame, max_depth: int, max_quiescence_depth: int, model: Model):
        self.game = game

        self.max_depth = max_depth
        self.max_quiescence_depth = max_quiescence_depth

        self.transposition_table = TranspositionTable()

        self.model = model

    def get_move(self) -> chess.Move:
        # get the current board
        board = self.game.get_board()
        # board_copy = chess.Board(board.fen())

        self.game.increment_move_counter()

        # get the opening book move
        opening_book_move = self.get_opening_book_move(board)
        # check if the opening book move is not None
        if opening_book_move is not None:
                return opening_book_move

        # get the best move
        alpha = -np.inf
        beta = np.inf

        best_value = -np.inf
        best_move = chess.Move.null()
        
        for depth in range(1, self.max_depth + 1):
            [move, value] = self.negamax_alpha_beta(board, alpha, beta, depth)

            if value > best_value:
                best_value = value
                best_move = move

        # return the best move
        print("best move: ", best_move, " color: ", board.turn)
        return move

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

        og_alpha = alpha

        tt_entry = self.transposition_table.get(board)
        
        if tt_entry is not None:
            if tt_entry.depth >= depth:
                if tt_entry.flag == "EXACT":
                    return [tt_entry.best_move, tt_entry.value]
                elif tt_entry.flag == "LOWER":
                    alpha = max(alpha, tt_entry.value)
                elif tt_entry.flag == "UPPER":
                    beta = min(beta, tt_entry.value)

            if alpha >= beta:
                return [tt_entry.best_move, tt_entry.value]
        
        if depth <= 0:
            # print("depth: ", depth, " move: ", best_move, " alpha: ", alpha, " beta: ", beta)
            return [best_move, self.quiescence_search(board, alpha, beta, self.max_quiescence_depth)]

        for move in board.legal_moves:
            # print("depth: ", depth, " move: ", move, " alpha: ", alpha, " beta: ", beta)
            board.push(move)
            board_value = -self.negamax_alpha_beta(board, -beta, -alpha, depth - 1)[-1]
            board.pop()

            if board_value >= beta:
                return [best_move, board_value] # fail soft beta-cutoff
                
            if board_value > max_value:
                max_value = board_value
                best_move = move

                if board_value > alpha:
                    alpha = board_value
            
            tt_depth = depth
            tt_value = max_value
            tt_best_move = best_move
            if max_value <= og_alpha:
                flag = "UPPER"
            elif max_value >= beta:
                flag = "LOWER"
            else:
                flag = "EXACT"

            self.transposition_table.put(board, tt_depth, tt_value, tt_best_move, flag)

        return [best_move, max_value]

    def quiescence_search(self, board, alpha, beta, depth):
        stand_pat_score = self.evaluation(board)
        # stand_pat_score = self.evaluation_model(board) # ML evaluation; similar to Stockfish

        if depth <= 0:
            return stand_pat_score
        
        if stand_pat_score >= beta:
            return beta
        
        if alpha < stand_pat_score:
            alpha = stand_pat_score
            
        for move in board.legal_moves:
            if board.is_capture(move):
                board.push(move)
                score = -self.quiescence_search(board, -beta, -alpha, depth - 1)
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

    def evaluation_model(self, board):
        fen_string = board.fen()
        fen_tensor = torch.tensor([fen_string])
        return self.model(fen_tensor)

class TranspositionTable():
    def __init__(self):
        self.size = (64, 12)
        self.table = np.zeros(self.size, dtype=np.int64)
        self.transposition_table = defaultdict(TTEntry)

        self.piece_constants = {
            (chess.PAWN, chess.WHITE): 0, 
            (chess.KNIGHT, chess.WHITE): 1, 
            (chess.BISHOP, chess.WHITE): 2, 
            (chess.ROOK, chess.WHITE): 3, 
            (chess.QUEEN, chess.WHITE): 4, 
            (chess.KING, chess.WHITE): 5, 
            (chess.PAWN, chess.BLACK): 6, 
            (chess.KNIGHT, chess.BLACK): 7, 
            (chess.BISHOP, chess.BLACK): 8, 
            (chess.ROOK, chess.BLACK): 9, 
            (chess.QUEEN, chess.BLACK): 10, 
            (chess.KING, chess.BLACK): 11
         }

        for i in range(64):
            for j in range(12):
                self.table[i, j] = int.from_bytes(uuid1().bytes, byteorder='big', signed=True) >> 64

    def get(self, board):
        key = self.zobrist_key(board)
        if key in self.transposition_table:
            return self.transposition_table[key]

        return None

    def put(self, board, depth, value, best_move, flag) -> None:
        key = self.zobrist_key(board)
        entry = TTEntry(key, depth, value, best_move, flag)
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
    def __init__(self, key: int, depth: int, value: float, best_move: chess.Move, flag: str):
        self.key = key
        self.depth = depth
        self.value = value
        self.best_move = best_move
        self.flag = flag


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
    ai = ChessAI(game, 4, 4, model_4_layer)
    
    board = game.get_board()
    count = game.get_move_counter()

    webbrowser.open("http://127.0.0.1:5000/")
    app.run()