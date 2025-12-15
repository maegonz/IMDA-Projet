import os
import glob
import torch
import pandas as pd
import random as rd
from typing import Tuple, List

def _extract_ids(file_pattern: str):
    """
    Extract unique (game_id, play_id) pairs from files matching pattern.
    
    Returns set of IDs of files processed.
    """
    ids = set()  # avoid recurring IDs
    for f in glob.glob(file_pattern):
        try:
            df = pd.read_csv(f)
            if 'game_id' in df.columns and 'play_id' in df.columns:
                pairs = df[['game_id', 'play_id']].drop_duplicates().values
                ids.update(map(tuple, pairs))
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")
    return ids

def get_game_play_ids(folder_path: str='./data/train'):
    """
    Extract unique game and play identifiers from input and output CSV files.

    Scans all input_*.csv and output_*.csv files in the folder,
    extracts unique (game_id, play_id) pairs from each, and returns
    them separately for inputs and outputs.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing input and output CSV files.

    Returns
    -------
    tuple[list, list]
        (input_ids, output_ids) where each is a sorted list of (game_id, play_id) tuples.
        Returns ([], []) if no files are found.

    Examples
    --------
    >>> input_ids, output_ids = get_game_play_ids('./data/train')
    >>> print(f"Input plays: {len(input_ids)}")
    >>> print(f"Output plays: {len(output_ids)}")
    """
    # Extract IDs from input and output files using patterns
    input_pattern = os.path.join(folder_path, "input_*.csv")
    output_pattern = os.path.join(folder_path, "output_*.csv")
    
    input_ids = _extract_ids(input_pattern)
    output_ids = _extract_ids(output_pattern)

    # Sort and convert to lists
    input_ids = sorted(input_ids)
    output_ids = sorted(output_ids)

    return input_ids, output_ids


def game_choice(input_ids: List[Tuple[int, int]], output_ids: List[Tuple[int, int]], random_state: int=None):
    """
    Randomly select a (game_id, play_id) pair present in both input and output IDs.

    Parameters
    ----------
    input_ids : list[tuple[int, int]]
        List of (game_id, play_id) tuples from input files.
    output_ids : list[tuple[int, int]]
        List of (game_id, play_id) tuples from output files.

    Returns
    -------
    tuple[int, int]
        A randomly selected (game_id, play_id) pair present in both lists.
    """
    if random_state is not None:
        rd.seed(random_state)  # for reproducibility

    assert set(input_ids) & set(output_ids), "No common (game_id, play_id) pairs found."
    common_ids = list(set(input_ids) & set(output_ids))
    game, play = rd.choice(common_ids)
    return game, play


def seed_everything(seed: int):
    """
    Set random seed for reproducibility across various libraries.

    Parameters
    ----------
    seed : int
        The seed value to set.
    """
    rd.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False