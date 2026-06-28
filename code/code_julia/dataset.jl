using CSV
using DataFrames

# Function that create a dataframe from a cv file
function create_dataset(csv::String)

    df = CSV.read(csv, DataFrame)

    return df
end