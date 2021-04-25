library(liger)
library(ggplot2)
library(cowplot)
theme_set(theme_cowplot())
library(Matrix)

args <- commandArgs(trailingOnly = TRUE)
spatialRNA.filename <- paste0('../spatial_scRNA_dat/', args[1])
# dataset col names should be cells, row names should be genes
scATAC <- read.csv("../scATAC_dat/promoter_genebody.GSM2668117_e11.5.nchrM.merge.sel_cell.mat.tsv",
                   header = T, row.names = 1)
scRNA <- readMM("../scRNA_dat/E10.5/filtered_gene_count.mtx.gz")

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

integrate.data <- list(scRNA=scRNA.common, scATAC=scATAC.common, spatialRNA=spatialRNA.common)
sdata <- createLiger(integrate.data)
# normalize data
sdata <- normalize(sdata)
# Can pass different var.thresh values to each dataset if one seems to be contributing significantly
# more genes than the other
sdata <- selectGenes(sdata, datasets.use=c(1,3))
sdata <- scaleNotCenter(sdata)

# k.suggest <- suggestK(sdata, num.cores = 32, nrep = 32)
# saveRDS(k.suggest, file = "3.dataset.default.K.RDS")
# lambda.suggest <- suggestLambda(sda, k.suggest, num.cores = 32, nrep = 32)
# saveRDS(lambda.suggest, file = "3.dataset.default.lambda.RDS")
# print(k.suggest)
# print(lambda.suggest)
sdata <- optimizeALS(sdata, k=20, lambda=5)
sdata <- quantile_norm(sdata)

saveRDS(sdata, file=paste0('../Result/promoter_genebody.', args[1],'.default.k.liger.RDS'))

library(reticulate)
use_python("~/anaconda3/envs/r4-base/bin/python")
use_condaenv(condaenv = "r4-base", conda = "~/anaconda3/bin/conda")
sdata <- louvainCluster(sdata, resolution = 0.25)
saveRDS(sdata, file=paste0('../Result/promoter_genebody.', args[1], '.default.k.liger.louvain.RDS'))
sdata <- runUMAP(sdata, distance = 'cosine')
saveRDS(sdata, file=paste0('../Result/promoter_genebody.', args[1], '.default.k.liger.umap.RDS'))

