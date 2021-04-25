library(liger)
library(ggplot2)
library(cowplot)
theme_set(theme_cowplot())
library(Matrix)

# dataset col names should be cells, row names should be genes
scRNA_PBMC_1 <- read.delim("../Batch_effect_data/batch_effect/dataset5/b1_exprs.txt",
                         header = T, row.names = 1)
scRNA_PBMC_2 <- read.delim("../Batch_effect_data/batch_effect/dataset5/b2_exprs.txt",
                         header = T, row.names = 1)

integrate.data <- list(PBMC_1=scRNA_PBMC_1, PBMC_2=scRNA_PBMC_2)
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

saveRDS(sdata, file=paste0('../Result/PBMC.default.k.liger.RDS'))

library(reticulate)
use_python("~/anaconda3/envs/r4-base/bin/python")
use_condaenv(condaenv = "r4-base", conda = "~/anaconda3/bin/conda")
sdata <- louvainCluster(sdata, resolution = 0.25)
saveRDS(sdata, file=paste0('../Result/PBMC.default.k.liger.louvain.RDS'))
sdata <- runUMAP(sdata, distance = 'cosine')
saveRDS(sdata, file=paste0('../Result/PBMC.default.k.liger.umap.RDS'))

