using Plots
using DataFrames

# Function that generate a graph of the predicted time series
# Require the original data fram ass well as the one with the predicted data
function plotForecasting(df_train::DataFrame, df_pred::DataFrame, labelReal::String, labelPred::String, titleGraph::String, filename::String)
    p = plot(df_train.globalEpisode, df_train.averageRating,
        label=labelReal)

    plot!(p, df_pred.globalEpisode, df_pred.averageRating,
        label=labelPred)

    xlabel!("Episode")
    ylabel!("Rating")
    title!(titleGraph)

    savefig(p, filename)
end