# Time series MLR iterative forecast
#

library(keras)
library(tidyverse)
library(dplyr)
library(wavelets)
library("data.table")

# Scaling
#
scale_data <- function(data) {
  return ((data - min(data)) / (max(data) - min(data)))
}

# Inverse scaling
#
unscale_data <- function(data,raw) {
  return (data * (max(raw) - min(raw)) + min(raw))
}

# Convert serie into dataset where endogen = s(t), exogen = {s(t-1),s(t-2),...,s(t-l)}
#
create_sequences <- function(data, lookback) {
  x <- list()
  y <- list()
  for (i in 1:(length(data) - lookback)) {
    x[[i]] <- data[i:(i + lookback - 1)]
    y[[i]] <- data[i + lookback]
  }
  return(list(x = do.call(rbind,x), y = do.call(rbind,y)))
}

# RMSE
#
rmse <- function(y,y_pred) {
  return(sqrt((mean(y-y_pred))^2))
}

# Extracting raw data
df <- read.csv("c:\\ml\\portfolio.csv")		# read raw data

rmse_mlr <- function(symbol,setplot=0) {

	dfi <- filter(df, stock == symbol)					# filter on parameter <symbol>
	dfi<-data.frame(date=as.Date(dfi$date),price=dfi$stockPrice)		# convert date
	data_raw<-dfi[,2]

	# remove last 4*7 days (to compare then actual data and forecast)
	#
	future_steps <- 4*7
	data_raw<-data_raw[1:(length(data_raw)-4*7)]

	# Filtering data_raw using Wavelets (optional)
	#
	mra<-mra(data_raw,filter="haar",n.levels=6)			# default is Daubechies
	data_ws3<-mra@S$S3[,1]						# extract level 3

	data <- scale_data(data_raw)
	#data <- scale_data(data_ws3)

	lookback <- 50
	dataset <- create_sequences(data, lookback)

	train_size <- length(dataset$y)					# no split train/test here, whole data used for training

	dt<-cbind(
		dx_train<-as.data.frame(x_train <- dataset$x[1:train_size,]),
		dy_train<-as.data.frame(y_train <- dataset$y[1:train_size])
	)
	colnames(dt)[1:50] <- paste("V",c(1:50), sep = "")
	colnames(dt)[51] <- "y_train"

	tt<-setDT(dt)

	# MLR model
	#
	model_mlr<-lm(y_train~
		V1+ V2+ V3+ V4+ V5+ V6+ V7+ V8+ V9+
		V10+V11+V12+V13+V14+V15+V16+V17+V18+V19+
		V20+V21+V22+V23+V24+V25+V26+V27+V28+V29+
		V30+V31+V32+V33+V34+V35+V36+V37+V38+V39+
		V40+V41+V42+V43+V44+V45+V46+V47+V48+V49+
		V50,
		data=tt) 

	# Forecast iterative
	#
	time_step <- lookback
	future_steps <- 4*7							# number of days to predict
	future_predictions <- c()

	current_seq <- data[(length(data) - time_step):length(data)]		# last sequence of actual data - lookback

	for (i in seq(future_steps)) {

  	    dw <- cbind(
		data.frame(array(current_seq, dim = c(1, lookback, 1))),	# convert serie to array, array to data frame
		data.frame(y_train=c(0))
  	    )
  	    colnames(dw)[1:50] <- paste("V",c(1:50), sep = "")			# name columns
  	    colnames(dw)[51] <- "y_train"
  	    tc<-setDT(dw)							# convert data frame to data table

  	    next_value <- model_mlr %>% predict(tc)				# predict next value for tc  
  	    n_next_value <- as.numeric(next_value[1])				# convert to number

  	    future_predictions <- c(future_predictions, n_next_value)		# append new prediction
  	    current_seq <- c(current_seq[-1], n_next_value)			# left shift current, inject prediction
	}
	data_extended<-c(data,future_predictions)

	if (setplot==1){
		png(filename=paste0("c:\\ml\\MLR_", symbol, ".png"))
		
		mi=min(data_raw[(length(data_raw)-4*7+1):(length(data_raw))],
			future_predictions * (max(data_raw) - min(data_raw)) + min(data_raw))
		ma=max(data_raw[(length(data_raw)-4*7+1):(length(data_raw))],
			future_predictions * (max(data_raw) - min(data_raw)) + min(data_raw))
		plot(data_raw[(length(data_raw)-4*7+1):(length(data_raw))],
			type='l',col='blue',ylim=c(mi,ma),ylab=paste0(symbol," Actual/Forecast"))
		points(future_predictions * (max(data_raw) - min(data_raw)) + min(data_raw),type='l',col='red')
		
		dev.off()
	}

	# Inverse scaling
	#
	data_base_extended <- unscale_data(data_extended, data_raw)

	r <- rmse(
		data_raw[(length(data_raw)-4*7+1):(length(data_raw))],
		future_predictions * (max(data_raw) - min(data_raw)) + min(data_raw))

	write.csv(data.frame(rmse=c(r)),paste0("c:\\ml\\MLR_", symbol, ".csv"),row.names = FALSE)

	return (r)
}

print(paste0("MLR Tesla:",	rmse_mlr("TSLA",1)))
print(paste0("MLR Google:", 	rmse_mlr("GOOGL",1)))
print(paste0("MLR Nvidia:", 	rmse_mlr("NVDA",1)))
print(paste0("MLR Apple:", 	rmse_mlr("AAPL",1)))
print(paste0("MLR Microsoft:",	rmse_mlr("MSFT",1)))
print(paste0("MLR Nokia:", 	rmse_mlr("NOK",1)))
print(paste0("MLR Rheinmetall:", rmse_mlr("RHM.F",1)))
print(paste0("MLR BAE Systems:", rmse_mlr("BA.L",1)))
print(paste0("MLR Dassault Av.", rmse_mlr("AM.PA",1)))
print(paste0("MLR Thalès:",	rmse_mlr("HO.PA",1)))
print(paste0("MLR Leonardo:",	rmse_mlr("FMNB.F",1)))

