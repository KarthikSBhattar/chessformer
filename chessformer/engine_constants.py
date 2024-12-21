# engines/constants.py

"""Constants for the engines."""

import functools
import os

import chess
import chess.engine
import chess.pgn
from jax import random as jrandom
import numpy as np

import chessformer.tokenizer as tokenizer
import chessformer.training_utils as training_utils
import chessformer.transformer as transformer
import chessformer.utils as utils
import chessformer.neural_engines as neural_engines


def _build_neural_engine(
    model_name: str,
    checkpoint_step: int = -1,
) -> neural_engines.NeuralEngine:
    """Returns a neural engine."""

    match model_name:
        case '9M':
            policy = 'action_value'
            num_layers = 8
            embedding_dim = 256
            num_heads = 8
        case '136M':
            policy = 'action_value'
            num_layers = 8
            embedding_dim = 1024
            num_heads = 8
        case '270M':
            policy = 'action_value'
            num_layers = 16
            embedding_dim = 1024
            num_heads = 8
        case 'local':
            policy = 'action_value'
            num_layers = 4
            embedding_dim = 64
            num_heads = 4
        case _:
            raise ValueError(f'Unknown model: {model_name}')

    num_return_buckets = 128

    match policy:
        case 'action_value':
            output_size = num_return_buckets
        case 'behavioral_cloning':
            output_size = utils.NUM_ACTIONS
        case 'state_value':
            output_size = num_return_buckets

    predictor_config = transformer.TransformerConfig(
        vocab_size=utils.NUM_ACTIONS,
        output_size=output_size,
        pos_encodings=transformer.PositionalEncodings.LEARNED,
        max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
        num_heads=num_heads,
        num_layers=num_layers,
        embedding_dim=embedding_dim,
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
    )

    predictor = transformer.build_transformer_predictor(config=predictor_config)
    checkpoint_dir = os.path.join(
        os.getcwd(),
        f'./checkpoints/{model_name}',
    )
    params = training_utils.load_parameters(
        checkpoint_dir=checkpoint_dir,
        params=predictor.initial_params(
            rng=jrandom.PRNGKey(1),
            targets=np.ones((1, 1), dtype=np.uint32),
        ),
        step=checkpoint_step,
    )
    _, return_buckets_values = utils.get_uniform_buckets_edges_values(
        num_return_buckets
    )
    return neural_engines.ENGINE_FROM_POLICY[policy](
        return_buckets_values=return_buckets_values,
        predict_fn=neural_engines.wrap_predict_fn(
            predictor=predictor,
            params=params,
            batch_size=1,
        ),
    )


ENGINE_BUILDERS = {
    'local': functools.partial(_build_neural_engine, model_name='local'),
    '9M': functools.partial(
        _build_neural_engine, model_name='9M', checkpoint_step=6_400_000
    ),
    '136M': functools.partial(
        _build_neural_engine, model_name='136M', checkpoint_step=6_400_000
    ),
    '270M': functools.partial(
        _build_neural_engine, model_name='270M', checkpoint_step=6_400_000
    ),
}
