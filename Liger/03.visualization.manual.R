library(liger)
library(ggplot2)
library(cowplot)
theme_set(theme_cowplot())
library(Matrix)
# dataset col names should be cells, row names should be genes
# args <- commandArgs(trailingOnly = TRUE)
args <- c()
args[1] <- 'GSM4096261_10t.tsv'
scATAC <- read.csv("../scATAC_dat/promoter_genebody.GSM2668117_e11.5.nchrM.merge.sel_cell.mat.tsv", header = T, row.names = 1)
scRNA <- readMM("../scRNA_dat/E10.5/filtered_gene_count.mtx.gz")
spatialRNA.filename <- paste0('../spatial_scRNA_dat/', args[1])
spatialRNA <- read.table(spatialRNA.filename, header=T, sep="\t", row.names = 1)
scRNA.rows <- read.csv("../scRNA_dat/E10.5/filtered_gene_annotate.csv.gz")
scRNA.cols <- read.csv("../scRNA_dat/E10.5/filtered_cell_annotate.csv.gz")
rownames(scRNA) <- scRNA.rows$gene_short_name
colnames(scRNA) <- scRNA.cols$sample

common_genes <- rownames(scRNA)[rownames(scRNA) %in% rownames(scATAC)]
common_genes <- common_genes[common_genes%in%rownames(spatialRNA)]
scATAC.common <- scATAC[rownames(scATAC) %in% common_genes,]
scRNA.common <- scRNA[rownames(scRNA) %in% common_genes,]
spatialRNA.common <- spatialRNA[rownames(spatialRNA) %in% common_genes,]
scATAC.common <- scATAC.common[match(common_genes, rownames(scATAC.common)),]
scRNA.common <- scRNA.common[match(common_genes, rownames(scRNA.common)),]
spatialRNA.common <- spatialRNA.common[match(common_genes, rownames(spatialRNA.common)),]

sdata <- readRDS(paste0('../Result/promoter_genebody.', args[1],'.default.k.liger.umap.RDS'))
all.tsne.coord <- as.data.frame(sdata@tsne.coords)
batch.label <- c(rep("scRNA", dim(scRNA.common)[2]), 
                 rep("scATAC", dim(scATAC.common)[2]), 
                 rep("spatialRNA", dim(spatialRNA.common)[2]))
all.tsne.coord <- cbind(all.tsne.coord, batch.label)
scRNA.annotation <- read.csv("../scRNA_dat/E10.5/filtered_cell_annotate.csv.gz", header = T, row.names = 1)
scRNA.celltype.standards <- read.table("../scRNA_dat/E10.5/kept_cell_types.txt", sep = "\t")
celltype.label <- c(scRNA.celltype.standards$V2[match(scRNA.annotation$Main_Cluster, scRNA.celltype.standards$V1)], 
                    rep(NA, dim(scATAC.common)[2]), 
                    rep(NA, dim(spatialRNA.common)[2]))
spatialRNA.coordinate <- matrix(NA, nrow = dim(spatialRNA)[2], ncol = 2)
for (i in 1:dim(spatialRNA)[2]) {
  spatialRNA.coordinate[i, 1] = strsplit(strsplit(colnames(spatialRNA.common)[i], split = "x")[[1]][1], split = "X")[[1]][2]
  spatialRNA.coordinate[i, 2] = strsplit(colnames(spatialRNA.common)[i], split = "x")[[1]][2]
}
cell.coordinate.X <- c(rep(NA, dim(scRNA.common)[2]), 
                       rep(NA, dim(scATAC.common)[2]), 
                       spatialRNA.coordinate[,1])
cell.coordinate.Y <- c(rep(NA, dim(scRNA.common)[2]), 
                       rep(NA, dim(scATAC.common)[2]), 
                       spatialRNA.coordinate[,2])
all.tsne.coord <- cbind(all.tsne.coord, celltype.label, cell.coordinate.X, cell.coordinate.Y)
tsne.plot.add.label <- function(tsne.coord, labelname, savefilename) {
  p <- ggplot(tsne.coord) +
    geom_point(aes(x=V1, y=V2, col=tsne.coord[,labelname], alpha=0.5)) +
    xlab("tsne_1") + ylab('tsne_2')
  ggsave(savefilename, plot = p, width = 25, height = 20)
}
tsne.plot.add.label(all.tsne.coord, "celltype.label", paste0(args[1], ".cell.type.pdf"))
tsne.plot.add.label(all.tsne.coord, "batch.label", paste0(args[1], ".batch.pdf"))
tsne.plot.add.label(all.tsne.coord, "cell.coordinate.X", paste0(args[1], ".cell.coordinate.X.pdf"))
tsne.plot.add.label(all.tsne.coord, "cell.coordinate.Y", paste0(args[1], ".cell.coordinate.Y.pdf"))


