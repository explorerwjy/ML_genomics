library(liger)
library(ggplot2)
library(cowplot)
theme_set(theme_cowplot())
library(Matrix)
library(Rfast)
# args <- commandArgs(trailingOnly = TRUE)
args <- c()
args[1] <- 'GSM4096261_10t.tsv'
sdata <- readRDS(paste0('../Result/promoter_genebody.', args[1],'.default.k.liger.umap.RDS'))
scRNA.annotation <- read.csv("../scRNA_dat/E10.5/filtered_cell_annotate.csv.gz", header = T, row.names = 1)
scATAC <- t(read.csv("../scATAC_dat/promoter_genebody.GSM2668117_e11.5.nchrM.merge.sel_cell.mat.tsv", header = T, row.names = 1))
spatialRNA.filename <- paste0('../spatial_scRNA_dat/', args[1])
spatialRNA <- read.table(spatialRNA.filename, header=T, sep="\t", row.names = 1)

predict.cell.type <- knn(sdata@H.norm[(dim(sdata@H$scRNA)[1]+dim(sdata@H$scATAC)[1]+1):dim(sdata@H.norm)[1],],
                         scRNA.annotation$Main_Cluster,
                         sdata@H.norm[1:dim(sdata@H$scRNA)[1], ],
                         10)
scRNA.celltype.standards <- read.table("../scRNA_dat/E10.5/kept_cell_types.txt", sep = "\t")
predict.cell.type <- scRNA.celltype.standards$V2[match(predict.cell.type, scRNA.celltype.standards$V1)]
spatialRNA.coordinate <- matrix(NA, nrow = dim(spatialRNA)[2], ncol = 2)
for (i in 1:dim(spatialRNA)[2]) {
  spatialRNA.coordinate[i, 1] = as.integer(strsplit(strsplit(colnames(spatialRNA.common)[i], split = "x")[[1]][1], split = "X")[[1]][2])
  spatialRNA.coordinate[i, 2] = as.integer(strsplit(colnames(spatialRNA.common)[i], split = "x")[[1]][2])
}
to.plot <- data.frame(X=spatialRNA.coordinate[,1],
                      Y=-spatialRNA.coordinate[,2],
                      predict.cell.type=predict.cell.type)
library(ComplexHeatmap)
library(RColorBrewer)
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_board = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))
my_colors = colorRampPalette(col_board[1:13])(13)
p <- ggplot(to.plot, aes(x=X, y=Y, col=predict.cell.type)) +
  geom_point(alpha=1) + scale_color_manual(values=my_colors)
ggsave(paste0(args[1], '.spatial.pdf'), plot = p, width = 7, height = 4.5)
write.csv(sdata@H.norm, paste0(args[1], '.cell-latent.csv'))
write.csv(sdata@W, paste0(args[1], '.latent-gene.csv'))


# predict.ATAC <- knn(sdata@H.norm[(dim(sdata@H$scRNA)[1]+dim(sdata@H$scATAC)[1]+1):dim(sdata@H.norm)[1],],
#                     scRNA.annotation$Main_Cluster,
#                     sdata@H.norm[(dim(sdata@H$scRNA)[1]+1):(dim(sdata@H$scRNA)[1]+dim(sdata@H$scATAC)[1]),],
#                     10)