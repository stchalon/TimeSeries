# Prophet
# CRAN: https://cran.r-project.org/web/packages/prophet/index.html
# Meta: https://facebook.github.io/prophet/docs/quick_start.html
# https://github.com/microprediction
#
#install.packages("prophet")
library(prophet)
library(tidyverse)
library(dplyr)

# RMSE
#
rmse <- function(y,y_pred) {
  return(sqrt((mean(y-y_pred))^2))
}

# Extracting raw data
df <- read.csv("c:\\ml\\portfolio.csv")		# read raw data

rmse_prophet <- function(symbol,setplot=0) {

	dfr <- filter(df, stock == symbol)				# filter on stock
	dfr <- data.frame(ds=as.Date(dfr$date),y=dfr$stockPrice)	# convert date

	# remove last 4*7 days (to compare then actual data and forecast)
	dfi <- dfr[1:(length(dfr$y)-4*7),]
	
	mi <- prophet(dfi)

	futurei <- make_future_dataframe(mi, periods = 4*7)
	forecast <- predict(mi, futurei)

	if (setplot==1){
		png(filename=paste0("c:\\ml\\Prophet_", symbol, ".png"))
		
		mi=min(dfr$y[(length(dfr$y)-4*7):(length(dfr$y))],
			forecast$yhat[(length(forecast$yhat)-4*7):(length(forecast$yhat))])
		ma=max(dfr$y[(length(dfr$y)-4*7):(length(dfr$y))],
			forecast$yhat[(length(forecast$yhat)-4*7):(length(forecast$yhat))])

		plot(dfr$y[(length(dfr$y)-4*7):(length(dfr$y))],
			type='l',col='blue',ylim=c(mi,ma),ylab=paste0(symbol," Actual/Forecast"))
		points(forecast$yhat[(length(forecast$yhat)-4*7):(length(forecast$yhat))],type='l',col='red')
		
		dev.off()
	}

	r <- rmse(
		dfr$y[(length(dfr$y)-4*7):(length(dfr$y))],
		forecast$yhat[(length(forecast$yhat)-4*7):(length(forecast$yhat))]
	)
	
	write.csv(data.frame(rmse=c(r)),paste0("c:\\ml\\Prophet_", symbol, ".csv"),row.names = FALSE)
	return (r)
}

print(paste0("Prophet Tesla:",	rmse_prophet("TSLA",1)))
print(paste0("Prophet Google:", rmse_prophet("GOOGL",1)))
print(paste0("Prophet Nvidia:", rmse_prophet("NVDA",1)))
print(paste0("Prophet Apple:", 	rmse_prophet("AAPL",1)))
print(paste0("Prophet Microsoft:",rmse_prophet("MSFT",1)))
print(paste0("Prophet Nokia:", 	rmse_prophet("NOK",1)))
print(paste0("Prophet Rheinmetall:", rmse_prophet("RHM.F",1)))
print(paste0("Prophet BAE Systems:", rmse_prophet("BA.L",1)))
print(paste0("Prophet Dassault Av.", rmse_prophet("AM.PA",1)))
print(paste0("Prophet Thalès:",	rmse_prophet("HO.PA",1)))
print(paste0("Prophet Leonardo:", rmse_prophet("FMNB.F",1)))

