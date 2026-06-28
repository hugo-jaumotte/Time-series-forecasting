import Pkg; Pkg.activate(@__DIR__)

using CSV
using DataFrames
using Plots
using HypothesisTests

include("dataset.jl")
include("tools.jl")
include("basicForecasting.jl")
include("arima.jl")
include("evaluationTests.jl")


# We create a graphe with the episodes on the x-axis and the ratings on the y-axis
df = create_dataset("../outputs/results/one_piece_clean.csv")
p = plot(df.globalEpisode, df.averageRating, label="Rating")
savefig(p, "../outputs/figures/jl_ep_ratings.pdf")


# We run statistic tests to define which forecasting model will be the most relevant to use
println("[Statistical tests] launching ...")
open("../outputs/results/stat_tests_results.txt", "w") do io
      println(io, "ADF test :")
      println(io,"")
      println(io, ADFTest(df.averageRating, :none, 1))
      println("[Statistical tests] ADF test (1/5) : done")
      println(io,"")
      println(io, ADFTest(df.averageRating, :constant, 1))
      println("[Statistical tests] ADF test (2/5) : done")
      println(io,"")
      println(io, ADFTest(df.averageRating, :trend, 1))
      println("[Statistical tests] ADF test (3/5) : done")
      println(io,"")
      println(io, ADFTest(df.averageRating, :constant, 2))
      println("[Statistical tests] ADF test (4/5) : done")
      println(io,"")
      println(io, ADFTest(df.averageRating, :constant, 4))
      println("[Statistical tests] ADF test (5/5) : done")

      println(io,"")
      println(io,"")

      println(io, "Ljung-Box test :")
      println(io,"")
      println(io, LjungBoxTest(df.averageRating, 2))
      println("[Statistical tests] Ljung-Box test (1/3) : done")
      println(io,"")
      println(io, LjungBoxTest(df.averageRating, 4))
      println("[Statistical tests] Ljung-Box test (2/3) : done")
      println(io,"")
      println(io, LjungBoxTest(df.averageRating, 8))
      println("[Statistical tests] Ljung-Box test (3/3) : done")
      println("")
end


# test of the models on training datasets
n = nrow(df)
halfsize = Int(floor(n / 2))

df_train = df[1:halfsize, :]
df_test  = df[halfsize+1:end, :]
df_train_size = nrow(df_test)


# Simple models : Naive, Average and Drift forecasting (training)
println("[Basic forecasting methods] launching ...")

# Naive forecasting
df_naive_pred = naiveForecasting(df_train, :globalEpisode, :averageRating, df_train_size)
CSV.write("../outputs/results/train_naive.csv", df_naive_pred)

plotForecasting(
      df_train, df_naive_pred,
      "Réel",
      "Prédiction (naive)",
      "Comparaison test vs prédiction (naive)",
      "../outputs/figures/jl_train_naive.pdf")

println("[Basic forecasting methods] Naive forecasting : done")

# Average forecasting
df_avg_pred = avgForcasting(df_train, :globalEpisode, :averageRating, df_train_size)
CSV.write("../outputs/results/train_avg.csv", df_avg_pred)

plotForecasting(
      df_train, df_avg_pred,
      "Réel",
      "Prédiction (mean)",
      "Comparaison test vs prédiction (mean)",
      "../outputs/figures/jl_train_avg.pdf")

println("[Basic forecasting methods] Average forecasting : done")

# Drift forecasting
df_drift_pred = driftForecasting(df_train, :globalEpisode, :averageRating, df_train_size)
CSV.write("../outputs/results/train_drift.csv", df_drift_pred)

plotForecasting(
      df_train, df_drift_pred,
      "Réel",
      "Prédiction (drift)",
      "Comparaison test vs prédiction (drift)",
      "../outputs/figures/jl_train_drift.pdf")

println("[Basic forecasting methods] Drift forecasting : done")
println("")


# Advanced model : ARIMA forecasting (training)
println("[ARIMA forecasting] launching ...")

# We run auto arima to find which are the best parameters for the arima model
df_auto_arima_pred, auto_arima_models = autoArimaForecasting(df_train, :globalEpisode, :averageRating, df_train_size)
CSV.write("../outputs/results/train_auto_arima.csv", df_auto_arima_pred)

plotForecasting(
      df_train, df_auto_arima_pred,
      "Réel",
      "Prédiction (auto ARIMA)",
      "Comparaison test vs prédiction (auto ARIMA)\n$(auto_arima_models[:averageRating])",
      "../outputs/figures/jl_train_auto_arima.pdf")

println("[ARIMA forecasting] Auto ARIMA {$(auto_arima_models[:averageRating])}: done")

# Even if we got a result from auto arima, we still decided to run others commmon arima model
orders = [
    (1, 1, 0),
    (1, 1, 1)
]

# We run the forecasting for each arima model and save the results (csv + graph)
for (p, d, q) in orders
      global df_arima_pred, models = arimaForecasting(df_train, :globalEpisode, :averageRating, df_train_size, p, d, q)
      CSV.write("../outputs/results/train_arima_$(p)_$(d)_$(q).csv", df_arima_pred)

      plotForecasting(
            df_train,
            df_arima_pred,
            "Réel",
            "Prédiction (ARIMA($p,$d,$q))",
            "Comparaison test vs prédiction (ARIMA($p,$d,$q))",
            "../outputs/figures/jl_train_arima_$(p)_$(d)_$(q).pdf"
      )
      println("[ARIMA forecasting] ARIMA($p,$d,$q) : done")
end
println("")


# Evaluation of the predictions
open("../outputs/results/models_evaluation.txt", "w") do io
      println("[Evaluation] MAE tests : launching ...")
      println(io, "Tests MAE,  models :")
      println(io, "Naive forecasting : $(mae(df_test.averageRating, df_naive_pred.averageRating))")
      println("[Evaluation] Tests MAE, naive : done")
      println(io, "Average forecasting : $(mae(df_test.averageRating, df_avg_pred.averageRating))")
      println("[Evaluation] Tests MAE, average : done")
      println(io, "Drift forecasting : $(mae(df_test.averageRating, df_drift_pred.averageRating))")
      println("[Evaluation] Tests MAE, drift : done")

      println(io,"")
      println(io,"")

      println(io, "Tests MAE, modèles ARIMA :")
      println(io, "Auto ARIMA forecasting : $(mae(df_test.averageRating, df_auto_arima_pred.averageRating))")
      println("[Evaluation] Tests MAE, auto ARIMA : done")
      for (p, d, q) in orders
            println(io, "ARIMA($p,$d,$q) forecasting : $(mae(df_test.averageRating, df_arima_pred.averageRating))")
            println("[Evaluation] Tests MAE, ARIMA($p,$d,$q) : done")
      end

      println(io,"")

      println("")
      println("[Evaluation] RMSE tests : launching ...")
      println(io, "Tests RMSE, modèles standards :")
      println(io, "Naive forecasting : $(rmse(df_test.averageRating, df_naive_pred.averageRating))")
      println("[Evaluation] Tests RMSE, naive : done")
      println(io, "Average forecasting : $(rmse(df_test.averageRating, df_avg_pred.averageRating))")
      println("[Evaluation] Tests RMSE, average : done")
      println(io, "Drift forecasting : $(rmse(df_test.averageRating, df_drift_pred.averageRating))")
      println("[Evaluation] Tests RMSE, drift : done")

      println(io,"")
      println(io,"")

      println(io, "Tests RMSE, modèles ARIMA :")
      println(io, "Auto ARIMA forecasting : $(rmse(df_test.averageRating, df_auto_arima_pred.averageRating))")
      println("[Evaluation] Tests RMSE, auto ARIMA : done")
      for (p, d, q) in orders
            println(io, "ARIMA($p,$d,$q) forecasting : $(rmse(df_test.averageRating, df_arima_pred.averageRating))")
            println("[Evaluation] Tests RMSE, ARIMA($p,$d,$q) : done")
      end
end
println("")


# Final prediction
println("[Final prediction] launching ...")
df_prediction, models = arimaForecasting(df, :globalEpisode, :averageRating, 8, 1, 0, 0)
      CSV.write("../outputs/results/final_prediction.csv", df_prediction)

      plotForecasting(
            df,
            df_prediction,
            "Réel",
            "Prédiction (ARIMA(1,0,0))",
            "Prédiction finale",
            "../outputs/figures/final_prediction.pdf"
      )
println("[Final prediction] done")

# Calc of the forecating average rating
println("[Prediction average rating] launching ...")
average_rating = mean(df_prediction.averageRating)
println("[Prediction average rating] average predicted rating : $average_rating")
println("[Prediction average rating] done")