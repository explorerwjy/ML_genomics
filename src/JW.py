import pandas as pd 
import numpy as np 
import csv
from os import walk
import numpy as np
import pandas as pd
import scipy
from scipy.stats import fisher_exact
from scipy.stats import binom_test
from scipy.stats import hypergeom
from matplotlib import pyplot as plt
import pickle

import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from sklearn.metrics import roc_curve, auc
import itertools

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import timeit
assert(torch.cuda.is_available()) # if this fails go to Runtime -> Change runtime type -> Set "Hardware Accelerator"

#https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE119945
#https://docs.google.com/document/d/1VrNAb1y8lWNuMQEw3I_AmZcQ-tDqfL9oBuVhGWBu8oo/edit

def human2mouse_gene():
	HMgene = pd.read_csv("../dat/HOM_MouseHumanSequence.rpt.txt", delimiter="\t")
	HomoloGenes = set(HMgene["HomoloGene ID"].values)
	Human2Mouse = {}
	Mouse2Human = {}
	for homo in HomoloGenes:
	    homo_df = HMgene[HMgene["HomoloGene ID"]==homo]
	    human_g = None
	    mouse_g = None
	    for i, row in homo_df.iterrows():
	        if row["Common Organism Name"] == "mouse, laboratory":
	            mouse_g = row["Symbol"]
	        if row["Common Organism Name"] == "human":
	            human_g = row["Symbol"]
	    if human_g != None and mouse_g != None:
	        Human2Mouse[human_g] = mouse_g
	        Mouse2Human[mouse_g] = human_g
	print(len(Human2Mouse))
	pickle.dump(Human2Mouse, open("data/human2mouse.map", 'wb'))
	pickle.dump(Mouse2Human, open("data/mouse2human.map", 'wb'))

def GenerateID2Grid():
    idx = 0
    dat = []
    for i in range(1, 51, 1):
        for j in range(1, 51, 1):
            dat.append([idx, "{}x{}".format(i, j)])
            idx += 1
    df = pd.DataFrame(data=dat, columns=["id", "grid"])
    df.to_csv("../Reformat_GSE137986/ID2Grid.tsv", index=False, sep="\t")
    return df

def reformat_GSE137986(fin, fout, ID):
    _ids = ID["grid"].values
    df = pd.read_csv(fin, delimiter="\t")
    print(df.shape)
    df.index = df["Unnamed: 0"].values
    dat = []
    for _id in _ids:
        try:
            dat.append([_id] + list(df.loc[_id, :].values[1:]))
        except:
            dat.append([_id] + list([np.nan] * (df.shape[1]-1)))
    df2 = pd.DataFrame(data=dat, columns=["ID"]+list(df.columns.values[1:]))
    df2.to_csv(fout, index=False, sep="\t")

def process():
    ID2Grid = GenerateID2Grid()
    DIR = "../spatial_RNA/GSE137986_RAW/"
    DIR2 = "../Reformat_GSE137986/"
    for (dirpath, dirnames, filenames) in walk(DIR):
        for file in filenames:
            if file.endswith(".tsv"):
                reformat_GSE137986(DIR+file, DIR2+file, ID2Grid)

def PlotGrid(grid_df):
    grids = np.zeros((50, 50))
    X = np.arange(50)
    Y = np.arange(50)
    for i in range(0, 50, 1):
        for j in range(0, 50, 1):
            grid = "{}x{}".format(i, j)
            if grid in grid_df.index:
                #ax.scatter(i, j, s=0.5)
                grids[i, j] = grid_df[grid]
    fig = plt.figure(dpi=200)
    ax = plt.subplot(111)
    ax.pcolormesh(X, Y, grids)
    fig.colorbar(c, ax=ax)

def ZscoreConverting(values):
    real_values = [x for x in values if x==x]
    mean = np.mean(real_values)
    std = np.std(real_values)
    zscores = []
    for x in values:
        z = (x - mean)/std
        zscores.append(z)
    return np.array(zscores)


class CNN(nn.Module):
    def __init__(self):
        """CNN Builder."""
        super().__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            #nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2),
            # Conv Layer block 2
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            #nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2),
            #nn.Dropout2d(p=0.2),
            # Conv Layer block 3
            #nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            #nn.BatchNorm2d(16),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=4, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(968, 10),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc_layer(x)
        return x


def spatial_transform(heads, oneD):
	grids = np.zeros((50, 50))
	for _col, _dat in zip(heads, oneD):
		x ,y = [int(x)-1 for x in _col.split("x")]
		grids[x, y] = _dat
	grids = grids[np.newaxis, :] 
	return torch.tensor(grids, dtype=torch.float32)

def generateFakeDat(real, N_fake, heads):
	dat = real[0][0] # get data matrix
	label = real[1]
	res = []
	for fake_xy in np.random.choice(heads, N_fake):
		x, y = [int(x)-1 for x in fake_xy.split("x")]
		tmp = np.zeros((50, 50))
		tmp[x,y] = 1
		tmp = dat + tmp
		tmp = tmp[np.newaxis, :] 
		tmp = torch.tensor(tmp, dtype=torch.float32)
		res.append((tmp, np.float32(label)))
	return res
		
def xxxx():
	xx = []
	for dat in training_dat_pos:
	    tmp = dat[0][0]
	    xx.append(np.count_nonzero(tmp)/ 1789)
	print(np.mean(xx))
	plt.hist(xx)
	plt.show()
	xx = []
	for dat in training_dat_neg:
	    tmp = dat[0][0]
	    xx.append(np.count_nonzero(tmp)/ 1789)
	print(np.mean(xx))
	plt.hist(xx)
	plt.show()

def getACC(model, loader, device):
	all_labels = np.array([])
	all_pred = np.array([])
	with torch.no_grad():
	    for data in loader:
	        inputs, labels = data[0].to(device), data[1].to(device)
	        outputs = model(inputs)
	        outputs = outputs.squeeze()
	        y_pred_tag = torch.sigmoid(outputs)
	        all_labels = np.concatenate((all_labels, labels.detach().cpu().numpy()))
	        all_pred = np.concatenate((all_pred, y_pred_tag.detach().cpu().numpy()))

	correct_results_sum1 = ((all_pred >= 0.5) ==  (all_labels ==1)).sum() / len(all_labels)
	correct_results_sum2 = ((all_pred < 0.5) ==  (all_labels ==0)).sum() / len(all_labels)
	#print(correct_results_sum1, correct_results_sum2)
	return all_labels, all_pred

def getACC_WGS(model, loader, device):
	all_labels = np.array([])
	all_pred = np.array([])
	with torch.no_grad():
	    for data in loader:
	        inputs, labels = data[0].to(device), data[1]
	        outputs = model(inputs)
	        outputs = outputs.squeeze()
	        y_pred_tag = torch.sigmoid(outputs)
	        all_labels = np.concatenate((all_labels, labels))
	        all_pred = np.concatenate((all_pred, y_pred_tag.detach().cpu().numpy()))

	correct_results_sum1 = ((all_pred >= 0.5) ==  (all_labels ==1)).sum() / len(all_labels)
	correct_results_sum2 = ((all_pred < 0.5) ==  (all_labels ==0)).sum() / len(all_labels)
	#print(correct_results_sum1, correct_results_sum2)
	return all_labels, all_pred