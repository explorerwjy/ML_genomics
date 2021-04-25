#%%
import numpy as np
import pandas as pd
import torch
from scipy.io import mmread
from scipy.sparse import csr_matrix
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter
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
            writer.add_scalar('Loss/train', loss.item(), counter)

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
#%%
if __name__ == '__main__':
    # ATAC
    scATAC = pd.read_csv('../scATAC_dat/promoter_genebody.GSM2668117_e11.5.nchrM.merge.sel_cell.mat.tsv', index_col=0)
    # Spatial
    spatialRNA = pd.read_csv("../spatial_scRNA_dat/GSM4189614_0713cL.tsv", sep='\t', index_col=0)
    # RNA
    filtered_gene_count = mmread("../scRNA_dat/filtered_gene_count.mtx.gz")
    filtered_gene_annotate = pd.read_csv("../scRNA_dat/filtered_gene_annotate.csv.gz")
    filtered_cell_annotate = pd.read_csv("../scRNA_dat/filtered_cell_annotate.csv.gz")
    rna_atac_merge = pd.merge(filtered_gene_annotate.reset_index(), scATAC.reset_index().reset_index(), left_on='gene_short_name', right_on='index', suffixes=('rna','atac'))
    rna_atac_spatial_merge = pd.merge(rna_atac_merge.iloc[:, 0:6], spatialRNA.reset_index().reset_index().iloc[:,0:2], left_on='gene_short_name', right_on='index', suffixes=('', 'spatial'))
#%%
    spatialRNA_dat = csr_matrix(spatialRNA.iloc[rna_atac_spatial_merge.level_0spatial.values,:].values.T, dtype=np.double)
    filtered_gene_count = csr_matrix(filtered_gene_count)
    scRNA_dat = csr_matrix(filtered_gene_count[rna_atac_spatial_merge.indexrna.values,:].T, dtype=np.double)
    scATAC_dat = scATAC.iloc[rna_atac_spatial_merge.level_0.values,]
    scATAC_dat = csr_matrix(scATAC.values.T, dtype=np.double)



    ncells = scATAC.shape[0]
    ngenes = scATAC.shape[1]
    print("Preprocessing DONE")


#%%
    # Initiate tensorboard
    writer = SummaryWriter()
    encoder = EncoderRNA(n_input=ngenes, n_output=20).cuda()
    decoder = MultiDecoderRNA(n_heads=3, n_input=20, n_output=ngenes).cuda()
#%%
# Load model if needed 
    encoder.load_state_dict(torch.load("encoder_spatial_brain.pt"))
    decoder.load_state_dict(torch.load("decoder_spatial_brain.pt"))
#%%
    train_dataset = DatasetRNA([scATAC_dat, scRNA_dat, spatialRNA_dat])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=400)

#%%
    optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), amsgrad=True)
    n_epoch = 5
    counter = 0
    epoch_losses = []
#%%    
    for i in range(n_epoch):
        epoch_loss, counter = run_one_epoch(counter, True, train_dataloader, encoder, decoder, optimizer, loss_lambda=5)
        print("Epoch: " + str(i) + " loss: " + str(epoch_loss))
        epoch_losses += epoch_loss
#%%
# Save model
torch.save(encoder.state_dict(), "encoder_spatial_brain.loss_lambda_5.pt")
torch.save(decoder.state_dict(), "decoder_spatial_brain.loss_lambda_5.pt")

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

all_cell_names = np.concatenate([scATAC.columns.values, filtered_cell_annotate['sample'].values, spatialRNA.columns.values])
#%%
shuffled_cell_names = all_cell_names[train_dataset.original_index]
# %%
latent_df = pd.DataFrame(latent, index=shuffled_cell_names)
latent_df['batches'] = batches
# %%
latent_df.to_csv("latent_df_spatial_brain.loss_lambda_5.csv")
# %%
