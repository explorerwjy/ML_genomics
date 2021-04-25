library(ggplot2)
library(cowplot)
theme_set(theme_cowplot())
library(Matrix)
library(Rfast)
# args <- commandArgs(trailingOnly = TRUE)
args <- c()
args[1] <- 'GSM4189614_0713cL.tsv'
latent <- as.matrix(read.csv('latent_df_spatial_brain.loss_lambda_5.csv', header = T, row.names = 1))
scRNA.annotation <- read.csv("../scRNA_dat/E10.5/filtered_cell_annotate.csv.gz", header = T, row.names = 1)
scATAC <- read.csv("../scATAC_raw/E11.5/E11.5.cell.tf.mat.csv", header = T, row.names = 1)
spatialRNA.filename <- paste0('../spatial_scRNA_dat/', args[1])
spatialRNA <- read.table(spatialRNA.filename, header=T, sep="\t", row.names = 1)
spatialRNA.cell.names <- colnames(spatialRNA)
for (i in 1:length(spatialRNA.cell.names)) {
  spatialRNA.cell.names[i] <- substr(spatialRNA.cell.names[i], 2, nchar(spatialRNA.cell.names[i]))
}
predict.cell.type <- knn(latent[match(spatialRNA.cell.names, rownames(latent)),],
                         scRNA.annotation$Main_Cluster,
                         latent[match(rownames(scRNA.annotation), rownames(latent)),],
                         10)
scRNA.celltype.standards <- read.table("../scRNA_dat/E10.5/kept_cell_types.txt", sep = "\t")
predict.cell.type <- scRNA.celltype.standards$V2[match(predict.cell.type, scRNA.celltype.standards$V1)]
spatialRNA.coordinate <- matrix(NA, nrow = dim(spatialRNA)[2], ncol = 2)
for (i in 1:dim(spatialRNA)[2]) {
  spatialRNA.coordinate[i, 1] = as.integer(strsplit(strsplit(colnames(spatialRNA)[i], split = "x")[[1]][1], split = "X")[[1]][2])
  spatialRNA.coordinate[i, 2] = as.integer(strsplit(colnames(spatialRNA)[i], split = "x")[[1]][2])
}
to.plot <- data.frame(X=spatialRNA.coordinate[,1],
                      Y=-spatialRNA.coordinate[,2],
                      predict.cell.type=predict.cell.type)
library(RColorBrewer)
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_board = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))
my_colors = colorRampPalette(col_board[1:13])(13)
p <- ggplot(to.plot, aes(x=X, y=Y, col=predict.cell.type)) +
  geom_point(alpha=1) + scale_color_manual(values=my_colors)
ggsave(paste0(args[1], '.spatial.pdf'), plot = p, width = 7, height = 4.5)

# ATAC
scATAC_cell_names <- read.delim('../scATAC_raw/GSM2668117_e11.5.nchrM.merge.sel_cell.xgi.txt.gz', header = F)
scATAC_mat <- as.matrix(read.csv("./E11.5.cell.tf.mat.csv", row.names = 1))
dir.create("ATAC.predict/")
for (i in 1:dim(scATAC_mat)[2]) {
  predict.ATAC <- knn(latent[match(spatialRNA.cell.names, rownames(latent)),],
                      scATAC_mat[,i],
                      latent[match(rownames(scATAC_mat), rownames(latent)),],
                      10)
  to.plot <- data.frame(X=spatialRNA.coordinate[,1],
                        Y=-spatialRNA.coordinate[,2],
                        predict.ATAC=log(predict.ATAC))
  mid<-(max(to.plot$predict.ATAC)+min(to.plot$predict.ATAC))/2
  p <- ggplot(to.plot, aes(x=X, y=Y, col=predict.ATAC)) +
    geom_point(alpha=1) + scale_color_gradient2(midpoint=mid, low="blue", mid="white",
                                                high="red", space ="Lab" )
  ggsave(paste0("ATAC.predict/", colnames(scATAC_mat)[i], '.spatial.pdf'), plot = p, width = 7, height = 4.5)
}
