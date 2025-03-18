# TimeGPT Nixtla
# Nixtla: https://www.nixtla.io/news/forecasting-in-r
# Nixtla API: https://dashboard.nixtla.io/1261136/api_keys
# API key nixak-KcUnCIXlmWHPKVJt2fpVjzE9Q3usl1HeyypFV7KqK6JoluIqJmCi5yVI3Cw9YLJMHFAE0nTsvtdXRqjq
# https://www.r-bloggers.com/2025/02/a-first-look-at-timegpt-using-nixtlar/
#

library(tidyverse)
library(dplyr)

library(nixtlar)
nixtla_set_api_key(api_key = "nixak-KcUnCIXlmWHPKVJt2fpVjzE9Q3usl1HeyypFV7KqK6JoluIqJmCi5yVI3Cw9YLJMHFAE0nTsvtdXRqjq")

# RMSE
#
rmse <- function(y,y_pred) {
  return(sqrt((mean(y-y_pred))^2))
}

# Extracting raw data
df <- read.csv("c:\\ml\\portfolio.csv")		# read raw data

rmse_nixtla <- function(symbol,setplot=0) {

	dfr <- filter(df, stock == symbol)				# filter on stock
	dfr <- data.frame(ds=as.Date(dfr$date),y=dfr$stockPrice)	# convert date

	# remove last 4*7 days (to compare then actual data and forecast)
	dfi <- dfr[1:(length(dfr$y)-4*7),]
	dfi<-cbind("unique_id",dfi)
	colnames(dfi)<-c("unique_id","ds","y")
	
	dfi_fcst<-nixtla_client_forecast(dfi, h = 4*7, level = c(80,95))

	dfi_extended<-c(dfi$y,dfi_fcst$TimeGPT)

	if (setplot==1){
		png(filename=paste0("c:\\ml\\TimeGPT_", symbol, ".png"))
		
		mi=min(dfr$y[(length(dfr$y)-4*7+1):(length(dfr$y))],
			dfi_fcst$TimeGPT)
		ma=max(dfr$y[(length(dfr$y)-4*7+1):(length(dfr$y))],
			dfi_fcst$TimeGPT)

		plot(dfr$y[(length(dfr$y)-4*7+1):(length(dfr$y))],
			type='l',col='blue',ylim=c(mi,ma),ylab=paste0(symbol," Actual/Forecast"))
		points(dfi_fcst$TimeGPT,type='l',col='red')
		
		dev.off()
	}

	r <- rmse(
		dfr$y[(length(dfr$y)-4*7+1):(length(dfr$y))],
		dfi_fcst$TimeGPT
	)
	
	write.csv(data.frame(rmse=c(r)),paste0("c:\\ml\\TimeGPT_", symbol, ".csv"),row.names = FALSE)

	return (r)
}

print(paste0("Nixtla TimeGPT Tesla:",	rmse_nixtla("TSLA",1)))
print(paste0("Nixtla TimeGPT Google:", rmse_nixtla("GOOGL",1)))
print(paste0("Nixtla TimeGPT Nvidia:", rmse_nixtla("NVDA",1)))
print(paste0("Nixtla TimeGPT Apple:", 	rmse_nixtla("AAPL",1)))
print(paste0("Nixtla TimeGPT Microsoft:",rmse_nixtla("MSFT",1)))
print(paste0("Nixtla TimeGPT Nokia:", 	rmse_nixtla("NOK",1)))
print(paste0("Nixtla TimeGPT Rheinmetall:", rmse_nixtla("RHM.F",1)))
print(paste0("Nixtla TimeGPT BAE Systems:", rmse_nixtla("BA.L",1)))
print(paste0("Nixtla TimeGPT Dassault Av.", rmse_nixtla("AM.PA",1)))
print(paste0("Nixtla TimeGPT Thalès:",	rmse_nixtla("HO.PA",1)))
print(paste0("Nixtla TimeGPT Leonardo:", rmse_nixtla("FMNB.F",1)))

