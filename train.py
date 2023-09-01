import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import TSDataset, causal_mask, prepare_time_series_data
from model import build_transformer

from config import get_config, get_weights_file_path

import pandas as pd

from torch.utils.tensorboard import SummaryWriter

import warnings
from tqdm import tqdm
from pathlib import Path

def greedy_decode(model, source, source_mask, decoder_in, scaler, device):

    # precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1,1,1).fill_(decoder_in.view(-1)[0]).type_as(source).to(device)
    for _ in range(decoder_in.shape[1]):

        # build a mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # print("Source:", source)
        # print("---shape:", source.shape)
        # print("Encoder output:", encoder_output)
        # print("---shape:", encoder_output.shape)
        # print("Decoder input:", decoder_input)
        # print("---shape:", decoder_input.shape)
        # print("Decoder mask:", decoder_mask)
        # print("---shape:", decoder_mask.shape)

        # calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get the next token
        pred = model.project(out)
        # print("Pred:", pred)

        pred_new = pred[:,-1,:]
        decoder_input = torch.cat([decoder_input, torch.empty(1,1,1).type_as(source).fill_(torch.tensor(scaler.transform(pred_new)).view(-1)[0]).to(device)], dim=1)
        # print("Decoder input_concat:", decoder_input)
        # print("---shape:", decoder_input.shape)
        # print(10*"-", "IT_END", 10*"-")

    # prediction "pred" should equal to "decoder_input[1:]", since there is only added the first initialization value
    return pred


def run_validation(model, validation_dataloader, scaler, device, print_msg, global_step, writer, epoch, num_examples=2):
    model.eval()
    count = 0

    src_input = []
    ground_truth = []
    predicted = []

    # size of the control window (just use a default value)
    # console_width = 80

    with torch.no_grad():
        batch_iterator_val = tqdm(validation_dataloader)
        for batch in batch_iterator_val:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_input = batch['decoder_input'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, decoder_input, scaler, device)

            src_data = batch["x_orig"][0]
            label = batch["label"][0]
            output = model_out.detach().cpu()

            src_input.append(src_data)
            ground_truth.append(label)
            predicted.append(output)
            
            # Print the source, target and model output
            # print_msg('-'*console_width)
            # print_msg(f"{f'SOURCE: ':>12}{src_data}")
            # print_msg(f"{f'TARGET: ':>12}{label}")
            # print_msg(f"{f'PREDICTED: ':>12}{output}")

            # if count == num_examples:
            #     txt_msg = '-'*int(console_width/2) + "END" + '-'*int(console_width/2)
            #     print_msg(txt_msg)
            #     break
        gt_torch = torch.stack(ground_truth)
        pred_torch = torch.stack(predicted)

        loss_fn = nn.MSELoss()
        loss = loss_fn(pred_torch.view(-1), gt_torch.view(-1))

    txt_msg = f"Validation loss of epoch {epoch}: {loss}"
    print_msg(txt_msg)

    return loss, ground_truth, predicted


def get_ds(config):
    ds_raw = pd.read_pickle(f'{config["path_pickle"]}/{config["data_pickle_name"]}')
    data_ts, scaler = prepare_time_series_data(ds_raw, config["exo_vars"], config["target"], config['src_seq_len'], config['tgt_seq_len'])

    # split 90-10 for training-validation
    train_ds_size = int(0.9 * len(data_ts))
    val_ds_size = len(data_ts) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(data_ts, [train_ds_size, val_ds_size])

    train_ds = TSDataset(train_ds_raw, config['src_seq_len'], config['tgt_seq_len'])
    val_ds = TSDataset(val_ds_raw, config['src_seq_len'], config['tgt_seq_len'])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    # return train_dataloader, val_dataloader
    return train_dataloader, val_dataloader, scaler

def get_model(config):
    model = build_transformer(len(config["exo_vars"] + config["target"]), len(config["target"]), config['src_seq_len'], config['tgt_seq_len'], d_model=config["d_model"], d_ff = config["d_ff"])
    return model

def train_model(config):
    # define the device on which we train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Make sure the weights folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, scaler = get_ds(config)
    model = get_model(config).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.MSELoss().to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch, seq_len, tgt_vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (batch, seq_len)

            # Compute the loss using MSE
            loss = loss_fn(proj_output.view(-1), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        run_validation(model, val_dataloader, scaler, device, lambda msg: batch_iterator.write(msg), global_step, writer, epoch)

        # Run validation at the end of every epoch
        # run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()

    train_model(config)