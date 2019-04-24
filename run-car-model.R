rm(list=ls())
setwd('~/Documents/GitHub/Spatial-image-analysis/')

library(fields,quietly=TRUE)
colfunc <- colorRampPalette(c("blue","lightblue","white","yellow","orange","red"))

library(CARBayes)
library(spdep)

source('lib/helper.R')

adj.matrix <- create.adj.matrix(28, 28)

handwritten.digit <- read.csv('images/flatten_handwritten_digit.csv', header=FALSE)
handwritten.digit <- t(handwritten.digit)[1,]
handwritten.digit <- expit(handwritten.digit)

fit.12 <- S.CARleroux(handwritten.digit ~ 1,
                      family='gaussian',
                      burnin=1000,
                      prior.mean.beta=c(0),
                      prior.var.beta=c(25),
                      W=adj.matrix,
                      prior.nu2=c(1, 0.01),
                      prior.tau2=c(1, 0.01),
                      rho=1, # no hetero effect
                      n.sample=11000)


#####################
### BREAST CANCER ###
#####################

# still takes way too long
# even though picture was scaled to 5% of its normal resolution

benign <- read.csv('images/benign.csv', header=FALSE)
x.dim.benign <- dim(benign)[1]
y.dim.benign <- dim(benign)[2]

adj.matrix.benign <- create.adj.matrix(x.dim.benign, y.dim.benign)

benign <- as.numeric(as.matrix(benign))

fit.benign <- S.CARleroux(expit(pad(benign, 1e-4)) ~ 1,
                          family='gaussian',
                          burnin=1000,
                          prior.mean.beta=c(0),
                          prior.var.beta=c(25),
                          W=adj.matrix.benign,
                          prior.nu2=c(1, 0.01),
                          prior.tau2=c(1, 0.01),
                          rho=1, # no hetero effect
                          n.sample=11000)

#####################
##### EXTENSION #####
#####################

adj.matrix <- create.adj.matrix(28, 28)
all.digits <- read.csv('images/all_digits.csv', header=FALSE)
parameter.means <- matrix(0, nrow=100, ncol=3)

# next bit takes a while
for (i in 1:100){
  # 10 first are 0's, then 10 next are 1's, etc.
  temp.fit <- S.CARleroux(expit(as.numeric(all.digits[i,])) ~ 1,
                          family='gaussian',
                          burnin=1000,
                          prior.mean.beta=c(0),
                          prior.var.beta=c(25),
                          W=adj.matrix,
                          prior.nu2=c(1, 0.01),
                          prior.tau2=c(1, 0.01),
                          rho=1, # no hetero effect
                          n.sample=11000)
  
  beta.mean <- mean(temp.fit$samples$beta)
  nu2.mean  <- mean(temp.fit$samples$nu2)
  tau2.mean <- mean(temp.fit$samples$tau2)
  
  parameter.means[i,] <- matrix(c(beta.mean,
                                nu2.mean,
                                tau2.mean),
                                ncol=1)
}

nice.plots <- function(result, col1, col2){
  '
  results should be parameters.means
  
  x axis: col1
  y axis: col2
  '
  
  cols <- c('#0000ff', # blue
            '#ffa500', # orange
            '#3cb371', # green
            '#ff0000', # red
            '#a020f0', # purple
            '#a0522d', # brown
            '#ff69b4', # pink
            '#696969', # grey
            '#6b8e23', # olive,
            '#00ffff') # cyan
  
  N <- length(result[,1])/10# number of samples for each digit
  
  minx <- min(result[,col1])
  maxx <- max(result[,col1])
  delta.x <- maxx - minx
  
  miny <- min(result[,col2])
  maxy <- max(result[,col2])
  delta.y <- maxy - miny
  
  xlim <- c(minx - delta.x*0.05, maxx + delta.x*0.05)
  ylim <- c(miny - delta.y*0.05, maxy + delta.y*0.05)
  
  plot(result[1:N, col1], 
       result[1:N, col2], 
       col=cols[1], 
       xlim=xlim,
       ylim=ylim,
       pch=20, # filling
       cex=2)  # dot size
  
  for (i in 2:10){
    par(new=FALSE)
    start <- 1 + N*(i-1)
    end   <- N*i
    plot(result[start:end, col1], 
          result[start: end, col2], 
          col=cols[i], 
          xlim=xlim,
          ylim=ylim,
          pch=20, # filling
          cex=2)  # dot size
  }
}

nice.plots(parameter.means, 2, 3)

