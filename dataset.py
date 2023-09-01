import pandas as pd

from typing import Any, List
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler


class TSDataset(Dataset):

    def __init__(self, ds: List[dict], src_seq_len: int, tgt_seq_len: int) -> None:
        super().__init__()
        self.ds = ds
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Any:
        datapoint = self.ds[index]
        enc_input = datapoint['x']
        dec_input = datapoint['y']
        label = datapoint['y_true']
        x_orig = datapoint["x_orig"]

        assert enc_input.size(0) == self.src_seq_len
        assert dec_input.size(0) == self.tgt_seq_len
        assert label.size(0) == self.tgt_seq_len

        return {
            "encoder_input": enc_input,  # (src_seq_len, n_features)
            "decoder_input": dec_input,  # (tgt_seq_len, n_tgt)
            "encoder_mask":  torch.ones(1, self.src_seq_len, self.src_seq_len).bool(), # (1, src_seq_len, src_seq_len), cannot be set to None because it is handled poorly by dataloader
            "decoder_mask": causal_mask(dec_input.size(0)), # (1, tgt_seq_len, tgt_seq_len),
            "label": label, # (tgt_seq_len, n_tgt)
            "x_orig": x_orig # (src_seq_len, n_features)
        }
    
    def collate_fn(self, batch):
        # Handle None values for encoder_mask
        encoder_mask = [item["encoder_mask"] for item in batch]
        encoder_mask = torch.stack(encoder_mask) if None not in encoder_mask else None
        
        # Stack other tensors
        other_tensors = {
            key: torch.stack([item[key] for item in batch]) for key in batch[0].keys() if key != "encoder_mask"
        }
        
        return {
            "encoder_input": other_tensors["encoder_input"],
            "decoder_input": other_tensors["decoder_input"],
            "encoder_mask": encoder_mask,
            "decoder_mask": other_tensors["decoder_mask"],
            "label": other_tensors["label"],
            "x_orig": other_tensors["x_orig"]
        }

# returns a triangular mask to protect the decoder to get information from future
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

def prepare_time_series_data(data: pd.DataFrame, exo_vars: list, target: list, input_seq_len: int, target_seq_len: int):
    data_array = data[target + exo_vars].values

    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_array)

    data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
    data_tensor_original = torch.tensor(data_array, dtype=torch.float32)

    data_seq = []

    num_obs = len(data_tensor)
    max_start_idx = num_obs - input_seq_len - target_seq_len

    for start_idx in range(max_start_idx):
        end_idx = start_idx + input_seq_len
        X_seq = data_tensor[start_idx:end_idx]
        x_orig = data_tensor_original[start_idx:end_idx, 0:len(target)]
        
        target_start_idx = end_idx - 1
        target_end_idx = target_start_idx + target_seq_len
        y_seq = data_tensor[target_start_idx:target_end_idx, 0:len(target)]
        y_seq_gt = data_tensor_original[target_start_idx+1:target_end_idx+1, 0:len(target)]

        data_seq.append({
        "x" : X_seq,
        "y" : y_seq,
        "y_true" : y_seq_gt,
        "x_orig" : x_orig,
        })

    return data_seq, scaler
