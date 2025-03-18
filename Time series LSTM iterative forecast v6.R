# Time series LSTM iterative forecast
# 

library(keras)
library(tidyverse)
library(dplyr)
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

rmse_lstm <- function(symbol,setplot=0) {

	# data filtering on <symbol> and dataset preparation
	#
	dfi <- filter(df, stock == "AAPL")					# filter on parameter <symbol>
	dfi<-data.frame(date=as.Date(dfi$date),price=dfi$stockPrice)		# convert date
	data_raw<-dfi[,2]

	# remove last 4*7 days (to compare then actual data and forecast)
	#
	future_steps <- 4*7
	data_raw<-data_raw[1:(length(data_raw)-4*7)]

	data <- scale_data(data_raw)

	lookback <- 50
	dataset <- create_sequences(data, lookback)

	# Model is trained on all available data (no split between training & test set)
	x_train <- dataset$x
	y_train <- dataset$y

	# Build LSTM model
	#
	model_lstm <- keras_model_sequential() %>%
  	  layer_lstm(units = 50, return_sequences = TRUE, input_shape = c(lookback, 1)) %>%
  	  layer_dropout(rate = 0.2) %>%
  	  layer_lstm(units = 50, return_sequences = FALSE) %>%
  	  layer_dropout(rate = 0.2) %>%
  	  layer_dense(units = 1)

	# Compile the model
	model_lstm %>% compile(
  	  loss = 'mean_squared_error',
  	  optimizer = optimizer_adam(),
  	  metrics = c('mae')
	)

	# Train the model
	history <- model_lstm %>% fit(
  	  x_train, y_train,
  	  epochs = 20,
  	  batch_size = 16,
  	  validation_split = 0.1,
  	  verbose = 0								# verbose=1 for log
	)

	# Forecast iterative
	#
	future_steps <- 4*7							# Number of values to predict
	future_predictions <- c()

	current_seq <- data[(length(dataset$y) - lookback):length(dataset$y)]	# last sequence of actual data

	for (i in seq(future_steps)) {

	    current_input <- array(current_seq, dim = c(1, lookback, 1))		# convert serie to array
	    next_value <- model_lstm %>% predict(current_input)
	    future_predictions <- c(future_predictions, next_value)
	    current_seq <- c(current_seq[-1], next_value)				# left shift current, inject prediction
	}

	# Inverse transform predictions
	future_predictions <- future_predictions * (max(data_raw) - min(data_raw)) + min(data_raw)

	if (setplot==1) {
		png(filename=paste0("c:\\ml\\LSTM_", symbol, ".png"))

		mi=min(data_raw[(length(data_raw)-4*7+1):(length(data_raw))],
			future_predictions )
		ma=max(data_raw[(length(data_raw)-4*7+1):(length(data_raw))],
			future_predictions )

		plot(data_raw[(length(data_raw)-4*7+1):(length(data_raw))],
			type='l',col='blue',ylim=c(mi,ma),ylab=paste0(symbol," Actual/Forecast"))
		points(future_predictions,type='l',col='red')

		dev.off()
	}

	r <- rmse(
		data_raw[(length(data_raw)-4*7+1):(length(data_raw))],
		future_predictions )

	write.csv(data.frame(rmse=c(r)),paste0("c:\\ml\\LSTM_", symbol, ".csv"),row.names = FALSE)

	return (r)
}

print(paste0("LSTM Tesla:",	rmse_lstm("TSLA",1)))
print(paste0("LSTM Google:", 	rmse_lstm("GOOGL",1)))
print(paste0("LSTM Nvidia:", 	rmse_lstm("NVDA",1)))
print(paste0("LSTM Apple:", 	rmse_lstm("AAPL",1)))
print(paste0("LSTM Microsoft:",	rmse_lstm("MSFT",1)))
print(paste0("LSTM Nokia:", 	rmse_lstm("NOK",1)))
print(paste0("LSTM Rheinmetall:", rmse_lstm("RHM.F",1)))
print(paste0("LSTM BAE Systems:", rmse_lstm("BA.L",1)))
print(paste0("LSTM Dassault Av.", rmse_lstm("AM.PA",1)))
print(paste0("LSTM Thalès:",	rmse_lstm("HO.PA",1)))
print(paste0("LSTM Leonardo:",	rmse_lstm("FMNB.F",1)))

