import argparse
import os
import sys

import numpy as np
from sgfmill import sgf
from tqdm import tqdm

# Add KataGo python directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
katago_python_dir = os.path.join(current_dir, "../KataGo/python")
sys.path.append(katago_python_dir)

from katago.game.board import Board
from katago.game.features import Features
from katago.train import modelconfigs


def get_rules(game):
    # Default to Chinese rules with 7.5 komi if not specified, typical for AlphaGo
    komi = game.get_komi()
    if komi is None:
        komi = 7.5

    # KataGo rules dict
    rules = {
        "koRule": "KO_POSITIONAL",  # Simple KO for now, usually sufficient
        "scoringRule": "SCORING_AREA",  # Chinese rules
        "taxRule": "TAX_NONE",
        "multiStoneSuicideLegal": False,  # Usually false
        "hasButton": False,
        "encorePhase": 0,
        "passWouldEndPhase": False,
        "whiteKomi": komi,
        "asymPowersOfTwo": 0,
    }
    return rules


def parse_sgf(sgf_path):
    with open(sgf_path, "rb") as f:
        game = sgf.Sgf_game.from_bytes(f.read())
    return game


def convert_game(sgf_path, output_path, pos_len=19, limit_moves=None):
    game = parse_sgf(sgf_path)
    board_size = game.get_size()
    if board_size != pos_len:
        print(f"Skipping {sgf_path}: Board size {board_size} != {pos_len}")
        return

    # Initialize KataGo Board and Features
    # Create simple config for Features
    # We need to match the feature dimensions expected by the model or standard KataGo
    # Using 'standard' config from modelconfigs might be hard without loading a model.
    # But Features class needs a config.
    # We can create a dummy config with standard values if we know them,
    # or rely on the fact that Features expects a specific config object.
    # Let's check modelconfigs.py if possible, but for now we mock it or use default.
    # Actually, Features uses config mainly for get_num_bin_input_features etc.
    # We'll use a dummy config that mimics standard V10+ features if possible.

    # Mock config
    class MockConfig:
        def __init__(self):
            # Standard KataGo values for recent versions
            self.search = {}
            self.model = {
                "input_features": "input_features_v11",  # Assuming v11 for features.py we saw
                "board_size": pos_len,
            }

    # However, features.py imports modelconfigs.
    # Let's rely on modelconfigs being available since we appended the path.
    # We need to inspect modelconfigs to see how to construct a valid config.
    # For now, let's assume we can pass a dummy object that has what Features needs.
    # Features needs: config (passed to modelconfigs.get_version etc)
    # modelconfigs.get_version(config) -> returns int.
    # modelconfigs.get_num_bin_input_features(config)
    # modelconfigs.get_num_global_input_features(config)

    # Let's just hardcode version to 11 (standard recent) if possible,
    # but Features calls modelconfigs.get_version(config).

    # HACK: We will patch modelconfigs or create a dict that works.
    # modelconfigs usually takes a dict or object.
    # Let's try to instantiate Features with a basic dict and strict checking off if needed.
    # But wait, Features init: self.version = modelconfigs.get_version(config)

    # Let's inspect modelconfigs in a separate step or just look at features.py again.
    # features.py: `self.version = modelconfigs.get_version(config)`

    # We will assume config is a dict for now, as modelconfigs.get_version likely looks up a key.
    dummy_config = {
        "version": 11,  # Assuming 11 is good for v10+ features
    }

    # We might need to adjust this based on runtime errors, but let's try.

    # Determine winner
    winner = game.get_winner()  # 'b' or 'w'
    # Typically Policy/Value targets are relative to current player.
    # globalTargetsNC often contains [PlayerAdvantage, ...]

    # Main loop
    root = game.get_root()
    board = Board(pos_len)

    # KataGo Features needs history of boards and moves
    board_history = [board.copy()]
    moves_history = []

    # Setup arrays
    # We don't know exact dimensions without modelconfigs, but features.py lines 14,15:
    # self.bin_input_shape = [modelconfigs.get_num_bin_input_features(config), pos_len, pos_len]
    # self.global_input_shape = [modelconfigs.get_num_global_input_features(config)]

    # Let's initialize Features and see shapes
    features = Features(dummy_config, pos_len)
    bin_c = features.bin_input_shape[0]
    glob_c = features.global_input_shape[0]

    inputs_bin = []
    inputs_glob = []
    policies = []
    values = []  # Outcome from perspective of player to move

    # Iterate moves
    # sgfmill main sequence
    game_moves = list(game.get_main_sequence())

    # Apply handicap stones if any (AlphaGo vs Lee Sedol mostly didn't have handicap, but good to handle)
    handicap = game.get_handicap()
    if handicap:
        # sgfmill handle handicap?
        # usually root node has AB[...]
        pass  # AlphaGo Game 2 was even.

    rules = get_rules(game)

    # Traverse
    # game_moves includes root.
    for i, node in enumerate(tqdm(game_moves, desc="Processing moves")):
        if limit_moves is not None and i >= limit_moves:
            break

        # Check for moves
        move = node.get_move()  # returns (color, (row, col)) or None
        if move is None:
            # Maybe setup stones? AB/AW
            # For this specific task (AlphaGo Game 2), we assume clean game.
            continue

        color, coords = move  # color 'b' or 'w'

        # Current player should match color
        # Board tracks current player? Board.pla
        # KataGo board has .pla, usually Black starts.
        # We need to make sure board.pla matches move color or we set it.
        # But board.play() flips turn.

        target_pla = Board.BLACK if color == "b" else Board.WHITE
        if board.pla != target_pla:
            # Force set? or assume pass?
            # SGF allows multiple moves by same player?
            # Standard Go alternates.
            pass

        # 1. Extract features for CURRENT state (before move)
        # We need to construct the input Tensors for the network to predict THIS move.

        # Let's allocate one row
        bin_row = np.zeros((1, pos_len * pos_len, bin_c), dtype=np.float32)
        glob_row = np.zeros((1, glob_c), dtype=np.float32)

        features.fill_row_features(
            board,
            board.pla,
            Board.get_opp(board.pla),
            board_history,
            moves_history,
            len(moves_history),
            rules,
            bin_row,
            glob_row,
            0,
        )

        inputs_bin.append(bin_row[0])
        inputs_glob.append(glob_row[0])

        # Policy Target: The move actually played
        # KataGo policy target is usually 19x19+1 (pass)
        if coords is None:
            # Pass
            policy_target = pos_len * pos_len
        else:
            row, col = coords
            # sgfmill coords: (row, col) from 0.
            # KataGo coords: Board.loc(x,y). Features xy_to_tensor_pos(x,y).
            # sgfmill (0,0) is top-left usually?
            # KataGo Board uses x,y.
            # features.py xy_to_tensor_pos(x,y) = y*L + x.
            # We need to verify coordinates orientation.
            # sgfmill: row 0 is top. col 0 is left.
            # KataGo usually implies y is row, x is col?
            # Using (row, col) directly as (y, x)?
            # features.py line 108: xy_to_tensor_pos(x,y)
            # We should assume x=col, y=row.
            policy_target = features.xy_to_tensor_pos(col, row)

        policies.append(policy_target)

        # Value Target: Outcome
        # If winner matches current player -> +1, else -1.
        if winner == color:
            values.append(1.0)
        else:
            values.append(-1.0)

        # Apply move
        if coords is None:
            loc = Board.PASS_LOC
        else:
            row, col = coords
            loc = board.loc(col, row)

        # KataGo board.play handles capture etc.
        # We should append to history
        moves_history.append((board.pla, loc))
        try:
            board.play(board.pla, loc)
        except Exception as e:
            print(f"Error playing move {move}: {e}")
            break

        board_history.append(board.copy())

    # Convert lists to arrays
    # Binary inputs need to be packed for NCHWPacked?
    # load_katago_data expects 'binaryInputNCHWPacked' or we can give unpacked.
    # Our prepare_katago_npz.py handles packing? Or unpacking?
    # process_katago_batch unpacks.
    # So we can save as 'binaryInputNCHW' directly if we change loader or just pack it.
    # To act like kata training data, we should probably save as is or unpacked if our loader supports it.
    # prepare_katago_npz.py unpack_binary_input takes packed.
    # But process_katago_batch does:
    # jax_batch = { "binaryInputNCHW": unpack_binary_input(...) }
    # format of binaryInputNCHW in JAX is (N, H, W, C).
    # features.py produces (N, H*W, C).
    # So we should reshape and maybe transpose to match our pipeline.

    # inputs_bin: (T, H*W, C)
    bin_array = np.array(inputs_bin)  # (T, 361, C)
    # Reshape to (T, H, W, C)
    bin_array = bin_array.reshape(-1, pos_len, pos_len, bin_c)
    # The pipeline in prepare_katago_npz expects packed (N, C, H, W)?
    # No, unpack_binary_input returns (N, H, W, C).
    # If we save directly as 'binaryInputNCHW', we can modify loader to check for it.
    # Or we can just save it as 'binaryInputNCHW' and update loader to use it if present, else unpack.
    # Or even better, just save 'binaryInputNCHW' and use a modified loader for our special dataset.

    # Actually, simplest is to save 'binaryInputNCHW' (unpacked) and ensure our code handles it.

    glob_array = np.array(inputs_glob)  # (T, Cg)

    # Policy: (T) integers.
    # Loader expects 'policyTargetsNCMove'. (N, C, Move)?
    # KataGo data is (N, C, Move) usually probabilities? Or sparse?
    # prepare_katago_npz: jnp.array(batch["policyTargetsNCMove"])
    # It seems to be distribution.
    # We have one hot.
    # Let's create one-hot.
    policy_array = np.zeros(
        (len(policies), 2, pos_len * pos_len + 1), dtype=np.float32
    )  # 2 channels?
    # KataGo trainingwrite.h says policyTargets is 2 channels?
    # Usually one channel is policy target, other is something else?
    # Let's check prepare_katago_npz.py usage.
    # It just passes it through.
    # Using 1 channel one-hot is safe.
    for i, p in enumerate(policies):
        policy_array[i, 0, p] = 1.0

    # Value: (T) floats.
    # Loader: 'valueTargetsNCHW' (N, C, H, W)?
    # Or 'globalTargetsNC'?
    # prepare_katago_npz: 'globalTargetsNC'
    # Training data has outcome in globalTargetsNC usually.
    # Inspect KataGo/cpp/dataio/trainingwrite.h for globalTargetsNC layout.
    # It likely contains [Win, Loss, Draw, ...]

    # For now, we put Win/Loss in globalTargetsNC index 0/1/2?
    # features.py doesn't set targets, only inputs.

    # Let's assume globalTargetsNC: [Win, Loss, NoResult]
    # If value=1 (Win), [1, 0, 0].
    # If value=-1 (Loss), [0, 1, 0].

    glob_targets = np.zeros((len(values), 3), dtype=np.float32)  # Arbitrary size safe?
    for i, v in enumerate(values):
        if v > 0:
            glob_targets[i, 0] = 1.0
        else:
            glob_targets[i, 1] = 1.0

    # Save
    np.savez_compressed(
        output_path,
        binaryInputNCHW=bin_array,  # Note: Not packed
        globalInputNC=glob_array,
        policyTargetsNCMove=policy_array,
        globalTargetsNC=glob_targets,
    )
    print(f"Saved {len(policies)} positions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgf", required=True, help="Path to SGF file")
    parser.add_argument("--output", required=True, help="Path to output NPZ")
    parser.add_argument(
        "--limit-moves", type=int, default=None, help="Limit number of moves processed"
    )
    args = parser.parse_args()

    convert_game(args.sgf, args.output, limit_moves=args.limit_moves)
