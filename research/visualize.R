library(ggplot2)
library(ggthemes)

getBest <- function(data){
	out <- NULL
	for(i in unique(data$idx)){
		data.sub <- subset(data, idx==i)
		df <- data.frame(model=c("0. PaloBoost", "1. SGTB-Bonsai", "2. XGBoost"),
				   		 value=c(max(data.sub[,1]), 
				   		 		max(data.sub[,2]),
				   		 		max(data.sub[,3])))
		out <- rbind(out, df)
	}
	return(out)
}
getLast <- function(data){
	out <- NULL
	for(i in unique(data$idx)){
		data.sub <- subset(data, idx==i)
		max.iter <- max(data.sub$nEst)
		data.sub <- subset(data.sub, nEst==max.iter)
		df <- data.frame(model=c("0. PaloBoost", "1. SGTB-Bonsai", "2. XGBoost"),
				   		 value=c(data.sub[1,1], 
				   		 		data.sub[1,2],
				   		 		data.sub[1,3]))
		out <- rbind(out, df)
	}
	return(out)	
}
getPerf <- function(data, mode){
	if(mode=="best"){
		return(getBest(data))
	}else{
		return(getLast(data))
	}
}

# Summary Plots

dataname <- "friedman"
ylab <- "R-squared"

dataname <- "hastie"
ylab <- "AUROC"

dataname <- "mort"
ylab <- "AUROC"

dataname <- "los"
ylab <- "R-squared"

dataname <- "ca6hr"
ylab <- "AUROC"

depth.lst <- c(5, 7)
lr.lst <- c(0.1, 0.5, 1.0)
param.df <- expand.grid(depth=depth.lst, 
						lr=lr.lst)
data <- NULL
for(i in 1:nrow(param.df)){
	p <- param.df[i,]
	fn <- paste0("results/",
					dataname,
					"_500", 
					"_", sprintf("%.1f", p$lr), 
					"_", p$depth, 
					"_0.7.csv")
	d.raw <- read.csv(fn)
	for(mode in c("best", "last")){
		d.perf <- getPerf(d.raw, mode)
		d.perf$lr <- p$lr
		d.perf$depth <- paste0("tree.depth=",p$depth)
		d.perf$mode <- mode
		data <- rbind(data, d.perf)
	}
}

ggplot(data, aes(x=as.factor(lr), y=value, fill=model)) + 
geom_boxplot() + 
facet_grid(depth~mode) + 
theme_gdocs() + 
scale_fill_few() + 
ylab(ylab) + 
xlab("Learning Rate")

fn_out <- paste0("results/", dataname, "_results.png")
ggsave(fn_out, width=8, height=6)


# Iteration x Performance curves

dataname <- "friedman"
ylab <- "R-squared"

dataname <- "hastie"
ylab <- "AUROC"

depth.lst <- c(5, 7)
lr.lst <- c(0.1, 0.5, 1.0)
param.df <- expand.grid(depth=depth.lst, 
						lr=lr.lst)
data <- NULL
for(i in 1:nrow(param.df)){
	p <- param.df[i,]
	fn <- paste0("results/",
					dataname,
					"_500", 
					"_", sprintf("%.1f", p$lr), 
					"_", p$depth, 
					"_0.7.csv")
	d.raw <- read.csv(fn)
	d.raw <- subset(d.raw, idx==0)
	d.raw$depth <- paste0("tree.depth=",p$depth)
	d.raw$lr <- paste0("learning.rate=",p$lr)
	model.lst <- c("0. Paloboost", "1. SGTB-Bonsai", "2. XGBoost")
	for(j in 1:3){
		d.model <- d.raw[,c(j,4,5,6,7)]
		d.model$model <- model.lst[j]
		colnames(d.model) <- c("value", "nEst", "idx", "depth", "lr", "model")
		data <- rbind(data, d.model)
	}
}

ggplot(data, aes(x=nEst, y=value, colour=model, linetype=model)) + 
geom_line(size=1) + 
facet_grid(depth~as.factor(lr)) + 
theme_gdocs() + 
scale_fill_few() + 
ylab(ylab) + 
xlab("n_estimators")

fn_out <- paste0("results/", dataname, "_curves.png")
ggsave(fn_out, width=14, height=6)