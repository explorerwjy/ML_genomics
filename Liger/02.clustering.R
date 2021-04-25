library(liger)
library(ggplot2)
library(cowplot)
theme_set(theme_cowplot())
library(Matrix)

args <- commandArgs(trailingOnly = TRUE)
sdata <- readRDS(paste0('../Result/promoter_genebody.', args[1],'.default.k.liger.RDS'))
library(reticulate)
use_python("~/anaconda3/envs/r4-base/bin/python")
use_condaenv(condaenv = "r4-base", conda = "~/anaconda3/bin/conda")
sdata <- louvainCluster(sdata, resolution = 0.25)
saveRDS(sdata, file=paste0('../Result/promoter_genebody.', args[1], '.default.k.liger.louvain.RDS'))
sdata <- runUMAP(sdata, distance = 'cosine')
saveRDS(sdata, file=paste0('../Result/promoter_genebody.', args[1], 'default.k.liger.umap.RDS'))

pdf(paste0('../Result/promoter_genebody.', args[1], '.default.k.liger.umap.pdf'))
all.plots <- plotByDatasetAndCluster(sdata, axis.labels = c('UMAP 1', 'UMAP 2'), return.plots = T)
all.plots[[1]] + all.plots[[2]]
dev.off()

pdf(paste0('../Result/promoter_genebody.', args[1], '.default.k.liger.gene.loadings.pdf'))
gene_loadings <- plotGeneLoadings(sdata, do.spec.plot = FALSE, return.plots = TRUE)
dev.off()

# write.table(
#   as.data.frame(sdata@reductions$iNMF@cell.embeddings),
#   file=paste0('../Result/promoter_genebody.', args[1], '.default.k.liger.umap.csv'),
#   sep=',',
#   quote=FALSE,
# )