from module import *
from DataLoader import *
from torch import nn as nn
from scipy.sparse import csr_matrix, csc_matrix
from scipy.io import mmread, mmwrite
import pandas as pd
import numpy as np


def run_one_epoch(train_flag, dataloader, encoder, decoder, optimizer, loss_lambda=1, device="cuda"):
    torch.set_grad_enabled(train_flag)
    encoder.train() if train_flag else encoder.eval()
    decoder.train() if train_flag else decoder.eval()

    losses = []

    for (x, t) in dataloader:  # collection of tuples with iterator

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

        losses.append(loss.detach().cpu().numpy())

    return np.mean(losses)


if __name__ == '__main__':
    scATAC = pd.read_csv('../scATAC_dat/promoter_genebody.GSM2668117_e11.5.nchrM.merge.sel_cell.mat.tsv', index_col=0)
    ncells = scATAC.shape[1]
    ngenes = scATAC.shape[0]
    encoder = EncoderRNA(n_input=ngenes, n_output=10)
    encoder.to("cuda")
    decoder = MultiDecoderRNA(n_heads=2, n_input=10, n_output=ngenes)
    decoder.to("cuda")
    scATAC_split1 = csr_matrix(scATAC.values[:, :600].T, dtype=np.double)
    scATAC_split2 = csr_matrix(scATAC.values[:, 600:].T, dtype=np.double)
    train_dataset = DatasetRNA([scATAC_split1, scATAC_split2])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10)
    optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), amsgrad=True)
    run_one_epoch(True, train_dataloader, encoder, decoder, optimizer)




