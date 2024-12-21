import chessformer.engine_constants as engine_constants

engine = engine_constants._build_neural_engine(model_name="9M")

def play(board):
    try:
        move = engine.play(board)
        move_san = board.san(move)
        board.push(move)
        return move_san
    except:
        return "Unable to play move"