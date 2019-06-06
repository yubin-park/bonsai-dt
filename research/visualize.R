library(ggplot2)
library(ggthemes)

getBest <- function(data){
	out <- NULL
	for(m in unique(data$model)){
		for(i in unique(data$b_idx)){
			data.sub <- subset(data, b_idx==i & model==m)
			df <- data.frame(model=m, value=max(data.sub$value))
			out <- rbind(out, df)
		}
	}
	return(out)
}
getLast <- function(data){
	out <- NULL
	for(m in unique(data$model)){
		for(i in unique(data$b_idx)){
			data.sub <- subset(data, b_idx==i & model==m)
			max.iter <- max(data.sub$n_est)
			data.sub <- subset(data.sub, n_est==max.iter)
			df <- data.frame(model=m, value=data.sub$value[1])
			out <- rbind(out, df)
		}
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



data <- NULL
for(lr in c(0.1, 0.5, 1.0)){
	fn <- paste0("results/prune",
					"_", sprintf("%.1f", lr), 
					"_3.csv")
	d.raw <- read.csv(fn)
	d.raw$lr <- paste0("max.learning.rate=",lr)
	data <- rbind(data, d.raw)
}
data$prune_ratio <- (data$nodes_pre - data$nodes_post)/data$nodes_pre
ggplot(data, aes(x=iteration, y=prune_ratio)) + 
geom_line(alpha=0.5, size=0.1) + 
stat_smooth(method="loess", size=1) + 
facet_wrap(.~lr) + 
theme_minimal() + 
ylab("Prune Ratio") + 
xlab("Iterations")
fn_out <- "results/prune_ratio.png"
ggsave(fn_out, width=8, height=3)

data <- NULL
for(lr in c(0.1, 0.5, 1.0)){
	fn <- paste0("results/lr",
					"_", sprintf("%.1f", lr), 
					"_3.csv")
	d.raw <- read.csv(fn)
	d.raw$max.lr <- paste0("max.learning.rate=",lr)
	data <- rbind(data, d.raw)
}
ggplot(data, aes(x=iteration, y=lr)) + 
geom_line(alpha=0.5, size=0.1) + 
stat_smooth(method="loess", size=1) + 
facet_wrap(.~max.lr, scale="free") + 
theme_minimal() + 
ylab("Adjusted Learning Rate") + 
xlab("Iterations")
fn_out <- "results/learning_rate.png"
ggsave(fn_out, width=8, height=3)



# Summary Plots


# The palette with black:
cbPaletteS <- c("#a1dab4", "#41b5c4", "#2c7fb8")
cbPaletteQ <- c("#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3")
  

#dataname <- "friedman"
#ylab <- "R-squared"

#dataname <- "hastie"
#ylab <- "AUROC"

#dataname <- "los"
#ylab <- "R-squared"

#dataname <- "mort"
#ylab <- "AUROC"

dataname <- "ca6hr"
ylab <- "AUROC"

depth.lst <- c(3, 4, 5)
lr.lst <- c(0.1, 0.5, 1.0)
param.df <- expand.grid(depth=depth.lst, 
						lr=lr.lst)
data <- NULL
for(i in 1:nrow(param.df)){
	p <- param.df[i,]
	fn <- paste0("results/",
					dataname,
					"_200", 
					"_", sprintf("%.1f", p$lr), 
					"_", p$depth, 
					".csv")
	d.raw <- read.csv(fn)
	best <- getPerf(d.raw, "best")
	last <- getPerf(d.raw, "last")
	diff <- best$value - last$value
	d.perf <- data.frame(lr=p$lr, depth=as.factor(p$depth),
				model=best$model, best=best$value, last=last$value)
	data <- rbind(data, d.perf)
}
data$diff <- data$best-data$last

ggplot(data, aes(x=as.factor(lr), y=diff, fill=depth)) + 
geom_boxplot() + 
facet_grid(.~model) + 
theme_minimal() + 
theme(legend.position="top", legend.direction="horizontal") +
scale_fill_manual(values=cbPaletteS) + 
ylab("diff = best - last") + 
xlab("Learning Rate") 

fn_out <- paste0("results/", dataname, "_results.png")
ggsave(fn_out, width=8, height=3)


# Iteration x Performance curves
# Iteration x Performance curves
data <- NULL
for(i in 1:nrow(param.df)){
	p <- param.df[i,]
	fn <- paste0("results/",
					dataname,
					"_200", 
					"_", sprintf("%.1f", p$lr), 
					"_", p$depth, 
					".csv")
	d.raw <- read.csv(fn)
	d.raw$depth <- paste0("tree.depth=",p$depth)
	d.raw$lr <- paste0("max.learning.rate=",p$lr)
	data <- rbind(data, d.raw)
}

ggplot(data, aes(x=n_est, y=value, colour=model, linetype=model)) + 
stat_smooth(size=1.5, method="loess") + 
facet_grid(depth~as.factor(lr)) + 
theme_minimal() + 
scale_colour_manual(values=cbPaletteQ) + 
ylab(ylab) + 
xlab("n_estimators") #+ 
#coord_cartesian(ylim = c(0.54, 0.72))

fn_out <- paste0("results/", dataname, "_curves.png")
ggsave(fn_out, width=10, height=6)

if (dataname %in% c("mort", "ca6hr")){
	data <- NULL
	for(i in 1:nrow(param.df)){
		p <- param.df[i,]
		fn <- paste0("results/",
					dataname,
					"_200", 
					"_", sprintf("%.1f", p$lr), 
					"_", p$depth, 
					"_noised.csv")
		d.raw <- read.csv(fn)
		d.raw$depth <- paste0("tree.depth=",p$depth)
		d.raw$lr <- paste0("max.learning.rate=",p$lr)
		data <- rbind(data, d.raw)
	}

	ggplot(data, aes(x=n_est, y=value, colour=model, linetype=model)) + 
	stat_smooth(size=1.5, method="loess") + 
	facet_grid(depth~as.factor(lr)) + 
	theme_minimal() + 
	scale_colour_manual(values=cbPaletteQ) + 
	ylab(ylab) + 
	xlab("n_estimators") #+ 
    #coord_cartesian(ylim = c(0.54, 0.72))


	fn_out <- paste0("results/", dataname, "_curves_noised.png")
	ggsave(fn_out, width=10, height=6)
}