using Statistics

function mae(y_true, y_pred)
    return mean(abs.(y_true .- y_pred))
end

function rmse(y_true, y_pred)
    return sqrt(mean((y_true .- y_pred).^2))
end