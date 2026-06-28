using CSV, DataFrames
using Statistics

function naiveForecasting(dataset, refColumn, columns, size)
    if isa(columns, Symbol)
        columns = [columns]
    end
    time_step = dataset[2, refColumn] - dataset[1, refColumn]
    data = Dict{Symbol,Vector}(
        refColumn => [dataset[end, refColumn] + i * time_step for i in 1:size])
    for column in columns
        last_index = findlast(!ismissing, dataset[!, column])
        last_value = dataset[last_index, column]
        data[column] = fill(last_value, size)
    end
    return DataFrame(data)
end

function avgForcasting(dataset,refColumn,columns,size)
    if(isa(columns,Symbol))
        columns = [ columns ]
    end
    time_step = dataset[2,refColumn] - dataset[1,refColumn]
    data = Dict{Symbol,Vector}(refColumn => [dataset[end,
        refColumn]+i*time_step for i in 1:size])
    for column in columns
        data[column] = fill(mean(skipmissing(dataset[!,column])),size)
    end
    return DataFrame(data)
end

function driftForecasting(dataset, refColumn, columns, size)
    if isa(columns, Symbol)
        columns = [columns]
    end
    time_step = dataset[2, refColumn] - dataset[1, refColumn]
    data = Dict{Symbol,Vector}(
        refColumn => [dataset[end, refColumn] + i * time_step for i in 1:size]
    )
    for column in columns
        values = collect(skipmissing(dataset[!, column]))
        first_value = first(values)
        last_value  = last(values)
        n = length(values)
        drift = (last_value - first_value) / (n - 1)
        data[column] = [
            last_value + h * drift
            for h in 1:size
        ]
    end

    return DataFrame(data)
end