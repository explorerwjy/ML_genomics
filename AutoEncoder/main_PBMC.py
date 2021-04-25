#%%
import numpy as np
import pandas as pd
import torch
from scipy.io import mmread
from scipy.sparse import csr_matrix
from torch import nn as nn
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from DataLoader import *
from module import *


#%%
def run_one_epoch(counter, train_flag, dataloader, encoder, decoder, optimizer, loss_lambda=1, device="cuda"):
    torch.set_grad_enabled(train_flag)
    encoder.train() if train_flag else encoder.eval()
    decoder.train() if train_flag else decoder.eval()

    losses = []

    for i, (x, t) in tqdm(enumerate(dataloader)):  # collection of tuples with iterator
        
        x = x.float()
        t = t.int()

        (x, t) = (x.to(device), t.to(device))  # transfer data to GPU

        z = encoder(x)  # forward pass to hidden
        y_w, y_v, y = decoder(z, t)  # forward pass to output

        loss_fun = nn.MSELoss()
        loss = loss_fun(y, x) + torch.tensor(loss_lambda) * torch.norm(y_v) # numerically stable

        if train_flag:
            loss.backward()  # back propagation
            optimizer.step()
            optimizer.zero_grad()
            counter += 1
            #writer.add_scalar('Loss/train', loss.item(), counter)

        losses.append(loss.detach().cpu().numpy())

    return np.mean(losses), counter

def get_latent(dataloader, encoder, decoder, device="cuda"):
    torch.set_grad_enabled(False)
    encoder.eval()
    decoder.eval()
    zs = [] 
    ts = []
    for i, (x, t) in tqdm(enumerate(dataloader)):  # collection of tuples with iterator
        
        x = x.float()
        t = t.int()

        (x, t) = (x.to(device), t.to(device))  # transfer data to GPU

        z = encoder(x)  # forward pass to hidden
        zs.append(z.detach().cpu().numpy())
        ts += t

    return zs, ts

if __name__ == '__main__':
#%%
    # PBMC_1
    PBMC_1 = pd.read_csv('../PBMC/b1_exprs.txt.gz', index_col=0, sep="\t")
    # PBMC_2
    PBMC_2 = pd.read_csv('../PBMC/b2_exprs.txt.gz', index_col=0, sep="\t")
#%%
    genes_sum = PBMC_1.sum(axis=1)+PBMC_2.sum(axis=1)
    PBMC_1 = PBMC_1.loc[genes_sum != 0]
    PBMC_2 = PBMC_2.loc[genes_sum != 0]
    ncells = PBMC_1.shape[1]
    ngenes = PBMC_1.shape[0]
    PBMC_1_dat = csr_matrix(PBMC_1.values.T, dtype=np.double)
    PBMC_2_dat = csr_matrix(PBMC_2.values.T, dtype=np.double)
    print("Preprocessing DONE")


#%%
    # Initiate tensorboard
    # writer = SummaryWriter()
    encoder = EncoderRNA(n_input=ngenes, n_output=20)
    decoder = MultiDecoderRNA(n_heads=2, n_input=20, n_output=ngenes)
#%%
# Load model if needed 
#    encoder.load_state_dict(torch.load("encoder_spatial_brain.pt"))
#    decoder.load_state_dict(torch.load("decoder_spatial_brain.pt"))
#%%
    train_dataset = DatasetRNA([PBMC_1_dat, PBMC_2_dat])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=400)

#%%
    optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), amsgrad=True)
    n_epoch = 20
    counter = 0
    epoch_losses = []
#%%    
    for i in range(n_epoch):
        epoch_loss, counter = run_one_epoch(counter, True, train_dataloader, encoder, decoder, optimizer, loss_lambda=5)
        print("Epoch: " + str(i) + " loss: " + str(epoch_loss))
        epoch_losses += epoch_loss
#%%
# Save model
torch.save(encoder.state_dict(), "encoder_PBMC.latent.20.loss_lambda_5.pt")
torch.save(decoder.state_dict(), "decoder_PBMC.latent.20.loss_lambda_5.pt")

# %%
    # Acquire shared component for the whole dataset and same
latent, batches = get_latent(train_dataloader, encoder, decoder)
latent = np.concatenate(latent)
batches = np.array([b.item() for b in batches])
#%%
    # np.savez_compressed("latent_and_batches.npz", latent=latent, batches=batches)
#%%
    # Load using this
    # alignment = np.load("alignment_and_batches.npz")['alignment']
    # batches = np.load("alignment_and_batches.npz")['batches']
# %%

all_cell_names = np.concatenate([PBMC_1.columns.values, PBMC_2.columns.values])
#%%
shuffled_cell_names = all_cell_names[train_dataset.original_index]
# %%
latent_df = pd.DataFrame(latent, index=shuffled_cell_names)
latent_df['batches'] = batches
# %%
latent_df.to_csv("latent_df_PBMC.latent.20.loss_lambda_5.csv")
# %%
