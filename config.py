from pathlib import Path

def get_config():
    return {
        "data_pickle_name": "price_15PM_trans.pkl",
        "path_pickle": ".",
        "exo_vars": [],
        "target": ["system_imbalance"],
        "batch_size": 16,
        "num_epochs": 5,
        "lr": 10**-3,
        "src_seq_len": 8,
        "tgt_seq_len": 2,
        "d_model": 128,
        "d_ff": 512,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)