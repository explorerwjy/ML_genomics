spatial.RNA.plot.a.gene <- function(spatial.RNA.filename, genename, log.normalize=TRUE, savefilename="tmp.pdf") {
  library(ggplot2)
  library(ggthemes)
  library(ggeasy)
  spatialRNA <- read.table(spatial.RNA.filename,
                           header=T, sep="\t", row.names = 1)
  LogNormalize <- function(M, scale = 1e5, normalize = TRUE){
    if(normalize){
      MM <- apply(M, 2, function(x){
        if(sum(x) > 0){
          x/sum(x)
        }else{
          x
        }
      })
    } else {
      MM <- M
    }
    MM <- scale * MM
    MM <- log10(MM + 1)
  }
  if (!genename %in% row.names(spatialRNA)) {
    message(paste0(genename, " not in ", spatial.RNA.filename))
    return(NULL)
  } else {
    spatialRNA.coordinate <- matrix(NA, nrow = dim(spatialRNA)[2], ncol = 2)
    for (i in 1:dim(spatialRNA)[2]) {
      spatialRNA.coordinate[i, 1] = as.integer(strsplit(strsplit(colnames(spatialRNA)[i], split = "x")[[1]][1], split = "X")[[1]][2])
      spatialRNA.coordinate[i, 2] = as.integer(strsplit(colnames(spatialRNA)[i], split = "x")[[1]][2])
    }
    # first log normalize, if necessary
    if (log.normalize) {
      spatialRNA <- LogNormalize(spatialRNA)
    }
    to.plot <- data.frame(expression = spatialRNA[match(genename, rownames(spatialRNA)),],
                          X=spatialRNA.coordinate[,1],
                          Y=spatialRNA.coordinate[,2])
    mid<-1/2*(min(to.plot$expression)+max(to.plot$expression))
    p <- ggplot(to.plot, aes(x=X, y=Y, col=expression)) +
      geom_point(alpha=0.5, ) + 
      xlab("X") + ylab("Y") + 
      ggtitle(genename) + theme_classic() + ggeasy::easy_center_title()
      scale_color_gradient2(midpoint=mid, low="blue", mid="white", high="red")
    ggsave(savefilename)
  }
}