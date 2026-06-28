using CSV, DataFrames
using Statistics
using StateSpaceModels

function arimaForecasting(dataset, refColumn, columns, size, p, d, q)
    if isa(columns, Symbol)
        columns = [columns]
    end
    time_step = dataset[2, refColumn] - dataset[1, refColumn]
    data = Dict{Symbol, Vector}(
        refColumn => [dataset[end, refColumn] + i * time_step for i in 1:size])
    models = Dict{Symbol, Any}()
    for column in columns
        series = collect(skipmissing(dataset[!, column]))
        model = SARIMA(series; order=(p, d, q), include_mean=true)
        fit!(model; save_hyperparameter_distribution=false)
        forecast_result = forecast(model, size)
        data[column] = only.(forecast_result.expected_value)
        models[column] = model
    end

    return DataFrame(data), models
end

function autoArimaForecasting(dataset, refColumn, columns, size)
    if isa(columns, Symbol)
        columns = [columns]
    end
    time_step = dataset[2, refColumn] - dataset[1, refColumn]
    data = Dict{Symbol, Vector}(
        refColumn => [dataset[end, refColumn] + i * time_step for i in 1:size])
    models = Dict{Symbol, Any}()
    for column in columns
        series = collect(skipmissing(dataset[!, column]))
        model = auto_arima(series)
        forecast_result = forecast(model, size)
        data[column] = only.(forecast_result.expected_value)
        models[column] = model
    end

    return DataFrame(data), models
end