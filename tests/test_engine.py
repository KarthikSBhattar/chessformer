from chessformer import engine
import chess

def test_engine():
    board = chess.Board()
    print(engine.play(board))