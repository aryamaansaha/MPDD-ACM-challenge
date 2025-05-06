from dataclasses import dataclass
import json
from typing import List, Optional


@dataclass
class Config:
    data_root_path: str
    window_split_time: int = 1
    audio_feature_method: str = "wav2vec"
    video_feature_method: str = "openface"
    labelcount: int = 2
    track_option: str = "Track1"
    feature_max_len: int = 26
    batch_size: int = 2
    lr: float = 0.00002
    num_epochs: int = 200
    device: str = "mps"
    cv_folds: int = 5
    seed: int = 32

    # For Early MLP
    num_epochs_tuning: int = 15
    num_epochs_final: int = 50
    optuna_trials: int = 50
    model_save_path: str = './best_early_fusion_mlp.pth'
    audio_dim: Optional[int] = None 
    video_dim: Optional[int] = None 
    pers_dim: Optional[int] = None 
    num_classes: Optional[int] = None 

    # For Early LSTM
    gamma: int = 2
    alpha: float = 0.25

    # for transformers
    optuna_timeout: Optional[int] = 3600 
    class_weights: Optional[List[float]] = None 
    calculate_weights_dynamically: bool = True 
    embed_dim: int = 128               # Internal embedding dimension for Transformer encoders
    nhead: int = 4                     # Number of multi-head attention heads 
    num_encoder_layers: int = 2        # Number of stacked Transformer encoder layers
    dim_feedforward: int = 512         # Dimension of the feedforward network within Transformer layers
    fusion_hidden_dim: int = 128 

    @staticmethod
    def from_json(json_file_path: str) -> 'Config':
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        return Config(**data)