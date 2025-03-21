# Chronos, AutoGL
# https://arxiv.org/pdf/2403.07815
# https://huggingface.co/amazon/chronos-bolt-base
# https://analytics-zoo.readthedocs.io/en/latest/doc/Chronos/Overview/chronos.html
# https://medium.com/write-a-catalyst/zero-shot-time-series-forecasting-using-chronos-ad99240c8117
# https://www.youtube.com/watch?v=WxazoCVkBhg
#
# Chronos-Bolt is a family of pretrained time series forecasting models which can be used for zero-shot forecasting. It is based on the T5 encoder-decoder architecture and has been trained on nearly 100 billion time series observations. It chunks the historical time series context into patches of multiple observations, which are then input into the encoder. The decoder then uses these representations to directly generate quantile forecasts across multiple future steps—a method known as direct multi-step forecasting. Chronos-Bolt models are up to 250 times faster and 20 times more memory-efficient than the original Chronos models of the same size.
#
# in c:\\ML, run cmd
# python -m env chronos_env
# chronos_env\scripts\activate
# pip install chronos-forecasting autogl
#
library(reticulate)

use_virtualenv("c:/ml/chronos_env", required = TRUE)

chronos <- import("chronos")
torch <- import("torch")
pandas <- import("pandas")

# RMSE
#
rmse <- function(y,y_pred) {
  return(sqrt((mean(y-y_pred))^2))
}

# Extracting raw data
dfraw <- read.csv("c:\\ml\\portfolio.csv")		# read raw data

library(tidyverse)
library(dplyr)
rmse_chronos <- function(symbol,setplot=0) {

	dfr <- filter(dfraw, stock == symbol)				# filter on stock
	dfr <- data.frame(ds=as.Date(dfr$date),y=dfr$stockPrice)	# convert date

	# remove last 4*7 days (to compare then actual data and forecast)
	dfi <- dfr[1:(length(dfr$y)-4*7),]

	data <- dfi$y
	df <- pandas$DataFrame(dict(y=data))

	# Initialize the Chronos pipeline
	pipeline <- chronos$BaseChronosPipeline$from_pretrained(
		"amazon/chronos-t5-small",
		device_map = "cpu"
	)

	# Convert the R data to a torch sensor
	context <- torch$tensor(data)			# context$shape is torch.Size([1257])

	prediction_length <- 4*7
	forecast <- pipeline$predict(context = context, prediction_length = as.integer(prediction_length))

	# Convert the forecast to an R list
	forecast_r <- py_to_r(forecast)

	forecast_np <- forecast$cpu()$numpy()  # Convert to NumPy, > class(forecast_np) is "array" [1 20 28]
	forecast_r <- py_to_r(forecast_np)  # Now it's an R array

	dim(forecast_r)  # Check shape

	# Convert to a R matrix
	forecast_matrix <- matrix(forecast_r, nrow = 20, ncol = 28)


	############################
	dfi_extended<-c(dfi$y,forecast_matrix[ 1,])

	if (setplot==1){
		png(filename=paste0("c:\\ml\\ChronosAutoGL_", symbol, ".png"))
		
		mi=min(dfr$y[(length(dfr$y)-4*7+1):(length(dfr$y))],
			forecast_matrix)
		ma=max(dfr$y[(length(dfr$y)-4*7+1):(length(dfr$y))],
			forecast_matrix)

		pal <- colorRamp(c("red", "blue"))

		plot(dfr$y[(length(dfr$y)-4*7+1):(length(dfr$y))],
			type='l',col='black',ylim=c(mi,ma),ylab=paste0(symbol," Actual/Forecast"),lwd=2)

		for (x in 1:20) {
			points(forecast_matrix[ x,],type='l',col=pal( x/20))
		}	
		
		dev.off()
	}

	r <- rmse(
		dfr$y[(length(dfr$y)-4*7+1):(length(dfr$y))],
		forecast_matrix[ 1,]
	)
	
	write.csv(data.frame(rmse=c(r)),paste0("c:\\ml\\ChronosAutoGL_", symbol, ".csv"),row.names = FALSE)

	return (r)
}


print(paste0("Chronos AutoGL Tesla:",	rmse_chronos("TSLA",1)))
print(paste0("Chronos AutoGL Google:", rmse_chronos("GOOGL",1)))
print(paste0("Chronos AutoGL Nvidia:", rmse_chronos("NVDA",1)))
print(paste0("Chronos AutoGL Apple:", 	rmse_chronos("AAPL",1)))
print(paste0("Chronos AutoGL Microsoft:",rmse_chronos("MSFT",1)))
print(paste0("Chronos AutoGL Nokia:", 	rmse_chronos("NOK",1)))
print(paste0("Chronos AutoGL Rheinmetall:", rmse_chronos("RHM.F",1)))
print(paste0("Chronos AutoGL BAE Systems:", rmse_chronos("BA.L",1)))
print(paste0("Chronos AutoGL Dassault Av.", rmse_chronos("AM.PA",1)))
print(paste0("Chronos AutoGL Thalès:",	rmse_chronos("HO.PA",1)))
print(paste0("Chronos AutoGL Leonardo:", rmse_chronos("FMNB.F",1)))
