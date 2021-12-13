library(quantmod)
library('Metrics')
library('lsa')
library('neuralnet')
library(PerformanceAnalytics)
library(randomForest)
library(caret)
library(e1071)
library(mlbench)
library(randomForest)
library(boot)
library(caTools)
library(gam)
library(gbm)
library(xgboost)
library('quantmod')
library('xts')
library('forecast')
library(naivebayes)
library(MLmetrics)
library(tseries)
library(timeSeries)
library(forecast)
library(xts)
library(yardstick)
library(modeltime.gluonts)
library(tidymodels)
library(tidyverse)
library(timetk)



##ashley portfolio 
#Import Data
setwd('/Users/akshat113/stevens/fa800/')
returns = read.csv('returnData.csv')
head(returns)

# Build Minimum Variance Portfolio
library(quadprog)

returns <- returns[-1,] # remove first row of NAs
#rownames(returns) <- returns$Date # convert Date column into Index
#returns <- returns[,-1] # Remove date column

covMat <- cov(returns) # Covariance Matrix
n <- ncol(covMat)

# dVec = 0 since we are trying to minimize the variance of the portfolio
# meq = 1 since we only have one constraint

wmin <- -0.30 # minimum weight allowed for a single asset
wmax <- 0.30 # maximum weight allowed for a single asset

Amat = t(rbind(1,-diag(10), diag(10)))
dVec = dVev <- matrix(data = rep(0, times = 10), nrow = 10, ncol = 1)
bVec = c(1, -rep(wmax, 10), rep(wmin, 10)) # Constraints: Sum of weights = 1 & wmin =< weight >= wmax
meq = 1
mvPBt = solve.QP(Dmat = covMat, dvec = dVec, Amat = Amat, bvec = bVec, meq = meq)
mvPBt$solution

# Check to see that portfolio is fully invested:
sum(mvPBt$solution)

# Minimum Variance 
(minVar = t(mvPBt$solution)%*%covMat%*%mvPBt$solution)
#Volatility of Portfolio:
(Bt = sqrt(minVar))

# Weights of optimal portfolio:
sol1 <- as.data.frame(round(mvPBt$solution*100, digits = 2))
rownames(sol1) <- c("XOM_Return","SHW_Return","BA_Return","DUK_Return","UNH_Return","JPM_Return","AMZN_Return","KO_Return","AAPL_Return","AMT_Return")
colnames(sol1) <- c("Weight Invested (%)")
sol1

# Note, the weights above are rounded to two digits. The variance of the 
# portfolio is .000038 and the volatility is equal to 0.006203606


# Creation of portfolio with optimal weights:
weights = mvPBt$solution
weightedReturns = mapply('*', returns, weights)
portfolioReturns = rowSums(weightedReturns)

# Plot of portfolio returns:
timeframe = as.Date(rownames(returns))
plot(portfolioReturns, main = "Minimum Variance Portfolio - Returns", 
     type = 'l', x = timeframe)



#end of ashley code
timeframe <- as.Date(rownames(returns))
datedReturns <- data.frame(timeframe, portfolioReturns)
rownames(datedReturns) <- datedReturns$timeframe
datedReturns <- as.xts(datedReturns)
datedReturns = datedReturns[,c(-1)]
datedReturns$portfolioReturns = as.numeric(datedReturns$portfolioReturns)
datedReturns$portfolioReturns2 = as.numeric(datedReturns$portfolioReturns)
datedReturns = datedReturns[,c(-1)]



ret <- datedReturns
colnames(ret) <- c("daily.returns")
fin_df <- list()
fin_df2 <- data.frame(rep(0,220))
fin_df3 <- data.frame(rep(0,330))
data <- head(ret,nrow(ret)-66)
data1 = ret

for (i in c(2,5,10)){
  for (k in c(1,2,3)){
    fin_df <- list()
    train <- tail(head(data1,nrow(data1)-66),252*i)
    test <- head(tail(data1,66),22*k)
    
    
    train_10 = data.frame(date=index(train), coredata(train))
    train_10$id = "returns"
    test_1 = data.frame(date=index(test), coredata(test))
    test_1$id = "returns"
    
    actual_1 = data.frame(rbind(train_10,test_1))
    
    
    model_fit_deepar <- deep_ar(
      id                    = "id",
      freq                  = "D",
      prediction_length     = 22*k,
      lookback_length       = 48,
      epochs                = 5
    ) %>%
      set_engine("gluonts_deepar") %>%
      fit(daily.returns ~ ., train_10)
    
    model_fit_nbeats <- nbeats(
      id                    = "id",
      freq                  = "D",
      prediction_length     = 22*k,
      lookback_length       = 48,
      
      # Decrease EPOCHS for speed
      epochs                = 5,
      
      scale                 = TRUE
    ) %>%
      set_engine("gluonts_nbeats") %>%
      fit(daily.returns ~ ., train_10)
    
    
    # Model 4: auto_arima ----
    #model_fit_arima_no_boost <- arima_reg() %>%
    #  set_engine(engine = "auto_arima") %>%
    #  fit(daily.returns ~ ., train_10)
    #> frequency = 12 observations per 1 year
    
    models_tbl <- modeltime_table(
      model_fit_deepar,
      model_fit_nbeats
      #model_fit_arima_no_boost
    )
    # ---- CALIBRATE ----
    calibration_tbl <- models_tbl %>%
      modeltime_calibrate(new_data = test_1)
    
    # ---- ACCURACY ----
    calibration_tbl %>%
      modeltime_accuracy()
    
    # ---- FORECAST ----
    f = calibration_tbl %>%
      modeltime_forecast(
        new_data = test_1,
        actual_data = actual_1,
        conf_interval = 0.95
      )
    
    
    
    nested_data_tbl <- train_10 %>%
      
      # 1. Extending: We'll predict 52 weeks into the future.
      extend_timeseries(
        .id_var        = id,
        .date_var      = date,
        .length_future = 22*k
      ) %>%
      
      # 2. Nesting: We'll group by id, and create a future dataset
      #    that forecasts 52 weeks of extended data and
      #    an actual dataset that contains 104 weeks (2-years of data)
      nest_timeseries(
        .id_var        = id,
        .length_future = 22*k,
        .length_actual = 2768
      ) %>%
      
      # 3. Splitting: We'll take the actual data and create splits
      #    for accuracy and confidence interval estimation of 52 weeks (test)
      #    and the rest is training data
      split_nested_timeseries(
        .length_test = 22*k
      )
    
    rec_prophet <- recipe(daily.returns ~ date,  extract_nested_train_split(nested_data_tbl)) 
    
    wflw_prophet <- workflow() %>%
      add_model(
        prophet_reg("regression", seasonality_yearly = TRUE) %>% 
          set_engine("prophet")
      ) %>%
      add_recipe(rec_prophet)
    
    
    rec_xgb <- recipe(daily.returns ~ .,  extract_nested_train_split(nested_data_tbl)) %>%
      step_timeseries_signature(date) %>%
      step_rm(date) %>%
      step_zv(all_predictors()) %>%
      step_dummy(all_nominal_predictors(), one_hot = TRUE)
    
    
    wflw_xgb <- workflow() %>%
      add_model(boost_tree("regression") %>% set_engine("xgboost")) %>%
      add_recipe(rec_xgb)
    
    
    rec_ets <- recipe(daily.returns ~ date,  extract_nested_train_split(nested_data_tbl)) 
    model_spec_ets <- exp_smoothing() %>%
      set_engine(engine = "ets")
    
    wflw_ets <- workflow() %>%
      add_model(model_spec_ets) %>%
      add_recipe(rec_ets)
    
    rec_mars <- recipe(daily.returns ~ date,  extract_nested_train_split(nested_data_tbl))%>%
      step_date(date, ordinal = FALSE) %>%
      step_mutate(date_num = as.numeric(date)) %>%
      step_normalize(date_num) %>%
      step_rm(date)
    
    model_spec_mars <- mars(
      mode="regression"
    ) %>%
      set_engine("earth")
    
    wflw_marss <- workflow() %>%
      add_model(model_spec_mars) %>%
      add_recipe(rec_mars)
    
    
    
    nested_modeltime_tbl <- modeltime_nested_fit(
      # Nested data 
      nested_data = nested_data_tbl,
      
      # Add workflows
      wflw_prophet,
      wflw_xgb,
      wflw_ets
    )
    
    nested_modeltime_tbl %>% 
      extract_nested_test_accuracy() %>%
      table_modeltime_accuracy(.interactive = F)
    
    f2 = nested_modeltime_tbl %>% 
      extract_nested_test_forecast()
    
    print(i)
    fin_df =  c(fin_df,tail(f$.value,k*44),tail(f2$.value,k*66))
    fin_df.df <- do.call("rbind", lapply(fin_df, as.data.frame))
    write.csv(fin_df.df, file = paste("fin_df_",k,"_",i,".csv"))
    rm(model_fit_deepar,model_fit_nbeats,models_tbl,calibration_tbl,f,nested_modeltime_tbl,f2,fin_df)
    gc()
  }
}
