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

# Length of Stay 

depth.lst <- c(5, 7)
lr.lst <- c(0.1, 0.5, 1.0)
param.df <- expand.grid(depth=depth.lst, 
						lr=lr.lst)
data <- NULL
for(i in 1:nrow(param.df)){
	p <- param.df[i,]
	fn <- paste0("results/los_500", 
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
ylab("R-squared") + 
xlab("Learning Rate")
ggsave("results/los_results.png", width=8, height=6)

# Mortality
data <- NULL
for(i in 1:nrow(param.df)){
	p <- param.df[i,]
	fn <- paste0("results/mort_500", 
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
scale_colour_few() + 
scale_fill_few() + 
ylab("AUROC") + 
xlab("Learning Rate")
ggsave("results/mort_results.png", width=8, height=6)

# Cardiac Arrest 6 Hrs
data <- NULL
for(i in 1:nrow(param.df)){
	p <- param.df[i,]
	fn <- paste0("results/ca6hr_500", 
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
scale_colour_few() + 
scale_fill_few() + 
ylab("AUROC") + 
xlab("Learning Rate")
ggsave("results/ca6hr_results.png", width=8, height=6)



