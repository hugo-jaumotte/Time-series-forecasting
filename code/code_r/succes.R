# Succes rate definition
library(data.table)

# We load the raw dataframes
ratings <- fread("../data/title.ratings.tsv.gz")
episode <- fread("../data/title.episode.tsv.gz")

# “Join” of the two dataframes on tconst (IMDb ID)
episodes <- merge(
  ratings,
  episode[, .(tconst, parentTconst, seasonNumber, episodeNumber)],
  by = "tconst"
)

# We clean up by removing the overall ratings 
episodes <- episodes[!is.na(parentTconst)]
setorder(episodes, parentTconst, seasonNumber, episodeNumber)
fwrite(episodes, "../outputs/results/all_episodes_clean.csv")

# Defining the succes rate
threshold <- quantile(episodes$averageRating, 0.75)
threshold


# Visualisation
# Creation of a dataframe for one piece ep only
df_op <- episodes[episodes$parentTconst == "tt11737520", ]
setorder(df_op, seasonNumber, episodeNumber)
df_op[, globalEpisode := .I]
fwrite(df_op, "../outputs/results/one_piece_clean.csv")

# Rating per ep graph
pdf("../outputs/figures/r_ep_ratings.pdf")


plot(df_op$globalEpisode,
     df_op$averageRating,
     type = "l",
     main = "Rating par épisode (One Piece)",
     xlab = "Episode",
     ylab = "Rating")

dev.off()

# Histogram
pdf("../outputs/figures/r_histogramme.pdf")

hist(df_op$averageRating,
     main = "Distribution des ratings des épisodes",
     xlab = "Rating",
     breaks = 8)

dev.off()

# Autocorrelation Function
pdf("../outputs/figures/r_acf.pdf")

acf(df_op$averageRating,
    main = "Fonction d'autocorrelation")

dev.off()




# Descriptive Statistics
mean_rating <- mean(df_op$averageRating)
median_rating <- median(df_op$averageRating)
sd_rating <- sd(df_op$averageRating)
min_rating <- min(df_op$averageRating)
max_rating <- max(df_op$averageRating)

mean_rating
median_rating
sd_rating
min_rating
max_rating

