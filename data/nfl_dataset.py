"""NFL Dataset for player trajectory prediction using attention mechanism."""

import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Sequence, Tuple
from torch.utils.data import Dataset
from processing import HISTORY_SIZE, MAX_ACCEL, MAX_DIST, MAX_SPEED, get_angle_features

# File used to cache the fully built dataset
DEFAULT_DATASET_FILE = Path("./data/nfl_dataset.pt")
# Maximum number of context players kept (padding applied when fewer are present)
MAX_CONTEXT_PLAYERS = 22
# Number of features per context player
KEY_DIM = 9


class NFLDataset(Dataset):
    """Process play tracking CSVs into tensors for attention models.

    The dataset pairs historical player states (queries), surrounding context
    players (keys), and future displacements (labels) to train receiver position
    predictors.
    """
    def __init__(
        self,
        input_dir: str = "./data/train/",
        cache_file: Path | str = DEFAULT_DATASET_FILE,
        use_cache: bool = True,
        augment: bool = True,
    ):
        """Load or build the dataset.

        Parameters
        ----------
        input_dir : str, default "./data/train/"
            Directory containing matching `input_*.csv` and `output_*.csv` files.
        cache_file : Path or str, default DEFAULT_DATASET_FILE
            Location to cache preprocessed tensors for quicker reloads.
        use_cache : bool, default True
            Whether to read/write the cache instead of rebuilding every time.
        augment : bool, default True
            Apply vertical flip augmentation to each sample to double the dataset size.

        Raises
        ------
        ValueError
            If no input CSV files are found in ``input_dir`` or if preprocessing
            yields zero samples.
        """

        self.input_dir = Path(input_dir)
        self.cache_file = Path(cache_file)
        self.use_cache = use_cache
        self.augment = augment

        self.list_inputs = sorted(self.input_dir.glob("input_*.csv"))
        if not self.list_inputs:
            raise ValueError(f"No input files found in {self.input_dir}")

        if use_cache and self.cache_file.exists():
            self._load_cache()
        else:
            self.queries, self.keys, self.labels = self._build_dataset()
            if use_cache:
                self._save_cache()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        """Return one sample of query, keys, and label tensors.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Cloned tensors for the query sequence, context keys, and displacement label.
        """
        return (
            self.queries[idx].clone(),
            self.keys[idx].clone(),
            self.labels[idx].clone(),
        )

    def _load_cache(self):
        """Populate tensors from an existing cache file."""
        payload = torch.load(self.cache_file)
        self.queries = payload["queries"]
        self.keys = payload["keys"]
        self.labels = payload["labels"]

    def _save_cache(self):
        """Persist preprocessed tensors to the cache path."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"queries": self.queries, "keys": self.keys, "labels": self.labels},
            self.cache_file,
        )

    def _build_dataset(self):
        """Parse all input/output CSV pairs into tensors.

        Returns
        -------
        tuple of torch.Tensor
            Stacked tensors for queries, keys, and labels respectively.
        """
        samples = []  # List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

        for input_path in self.list_inputs:
            output_path = input_path.with_name(input_path.name.replace("input_", "output_"))
            if not output_path.exists():
                continue

            df_in = pd.read_csv(input_path)
            df_out = pd.read_csv(output_path)
            if df_in.empty or df_out.empty:
                continue

            common_plays = set(df_in["play_id"]).intersection(df_out["play_id"])
            for play_id in tqdm(common_plays, leave=False, desc=input_path.name):
                samples.extend(self._process_play(df_in, df_out, play_id))

        if not samples:
            raise ValueError("No samples were generated from the provided data.")

        queries = torch.stack([s[0] for s in samples])
        keys = torch.stack([s[1] for s in samples])
        labels = torch.stack([s[2] for s in samples])
        return queries, keys, labels

    def _process_play(
        self, df_in: pd.DataFrame, df_out: pd.DataFrame, play_id: int
    ):
        """Convert one play into multiple samples for each target player.

        Parameters
        ----------
        df_in : pandas.DataFrame
            Input tracking data containing historical frames.
        df_out : pandas.DataFrame
            Output tracking data containing future frames for labels.
        play_id : int
            Identifier of the play to process.

        Returns
        -------
        list of tuple
            Each tuple holds (query, keys, label) tensors for a target player instance.
        """
        play_in = (
            df_in[(df_in["game_id"] == df_in["game_id"].iloc[0]) & (df_in["play_id"] == play_id)]
            .sort_values("frame_id")
        )
        play_out = (
            df_out[(df_out["game_id"] == df_out["game_id"].iloc[0]) & (df_out["play_id"] == play_id)]
            .sort_values("frame_id")
        )

        candidates = play_in.loc[play_in["player_to_predict"].astype(bool), "nfl_id"].unique()
        samples = []  # List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

        for target_id in candidates:
            player_track = play_in[play_in["nfl_id"] == target_id]
            if len(player_track) < HISTORY_SIZE:
                continue

            history_seq = player_track.iloc[-HISTORY_SIZE:]
            current_pos = history_seq.iloc[-1]

            future_track = play_out[play_out["nfl_id"] == target_id]
            if future_track.empty:
                continue

            future_pos = future_track.iloc[-1]
            label = torch.tensor(
                [future_pos["x"] - current_pos["x"], future_pos["y"] - current_pos["y"]],
                dtype=torch.float32,
            )

            target_side = player_track["player_side"].iloc[0]
            ball_x, ball_y = current_pos["ball_land_x"], current_pos["ball_land_y"]
            query = self._build_query(history_seq, ball_x, ball_y, target_side == "Offense")
            keys = self._build_keys(play_in, current_pos, target_id, target_side)

            samples.append((query, keys, label))
            if self.augment:
                samples.append(self._flip_vertical(query, keys, label))

        return samples

    def _build_query(
        self,
        history_seq: pd.DataFrame,
        ball_x: float,
        ball_y: float,
        is_offense: bool,
    ):
        """Flatten receiver history into a single query vector.

        Parameters
        ----------
        history_seq : pandas.DataFrame
            Last ``HISTORY_SIZE`` frames for the target player.
        ball_x : float
            Ball landing x-coordinate for relative distance features.
        ball_y : float
            Ball landing y-coordinate for relative distance features.
        is_offense : bool
            Whether the target player is on offense (one-hot encoded in features).

        Returns
        -------
        torch.Tensor
            Query tensor of shape (HISTORY_SIZE * 9,).
        """
        rec_features: List[float] = []
        for _, row in history_seq.iterrows():
            sin_d, cos_d = get_angle_features(row["dir"])
            sin_o, cos_o = get_angle_features(row["o"])

            rec_features.extend(
                [
                    row["s"] / MAX_SPEED,
                    row["a"] / MAX_ACCEL,
                    sin_d,
                    cos_d,
                    sin_o,
                    cos_o,
                    (ball_x - row["x"]) / MAX_DIST,
                    (ball_y - row["y"]) / MAX_DIST,
                    1.0 if is_offense else 0.0,
                ]
            )

        return torch.tensor(rec_features, dtype=torch.float32)

    def _build_keys(
        self,
        play_in: pd.DataFrame,
        current_pos: pd.Series,
        target_id: int,
        target_side: str,
    ):
        """Build context keys for players on the same frame as the target.

        Parameters
        ----------
        play_in : pandas.DataFrame
            Full play input data.
        current_pos : pandas.Series
            Current frame row for the target player.
        target_id : int
            Target player's NFL identifier.
        target_side : str
            Offense/Defense side of the target player.

        Returns
        -------
        torch.Tensor
            Keys tensor of shape (MAX_CONTEXT_PLAYERS, KEY_DIM) with padding.
        """
        current_frame = current_pos["frame_id"]
        others = play_in[(play_in["frame_id"] == current_frame) & (play_in["nfl_id"] != target_id)]

        keys_list = []  # List[Sequence[float]]
        for _, other_row in others.iterrows():
            sin_d, cos_d = get_angle_features(other_row["dir"])
            sin_o, cos_o = get_angle_features(other_row["o"])

            keys_list.append(
                [
                    (other_row["x"] - current_pos["x"]) / MAX_DIST,
                    (other_row["y"] - current_pos["y"]) / MAX_DIST,
                    other_row["s"] / MAX_SPEED,
                    other_row["a"] / MAX_ACCEL,
                    sin_d,
                    cos_d,
                    sin_o,
                    cos_o,
                    1.0 if target_side == other_row["player_side"] else 0.0,
                ]
            )

        while len(keys_list) < MAX_CONTEXT_PLAYERS:
            keys_list.append([0.0] * KEY_DIM)

        return torch.tensor(keys_list[:MAX_CONTEXT_PLAYERS], dtype=torch.float32)

    def _flip_vertical(
        self, query: torch.Tensor, keys: torch.Tensor, label: torch.Tensor
    ):
        """Apply vertical flip augmentation to query, keys, and label.

        Parameters
        ----------
        query : torch.Tensor
            Original query tensor.
        keys : torch.Tensor
            Original keys tensor.
        label : torch.Tensor
            Original displacement label.

        Returns
        -------
        tuple of torch.Tensor
            Vertically flipped versions of (query, keys, label).
        """
        query_flip = query.clone()
        keys_flip = keys.clone()
        label_flip = label.clone()

        for i in range(HISTORY_SIZE):
            base = i * KEY_DIM
            query_flip[base + 3] *= -1  # cos_d
            query_flip[base + 5] *= -1  # cos_o
            query_flip[base + 7] *= -1  # dist_y

        keys_flip[:, 1] *= -1  # rel_y
        keys_flip[:, 5] *= -1  # cos_d
        keys_flip[:, 7] *= -1  # cos_o
        label_flip[1] *= -1

        return query_flip, keys_flip, label_flip