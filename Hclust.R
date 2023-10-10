# Load the required library
library(cluster)


df <- read.csv("CLEANED_HCLUST.csv")

--------------------
#Extract the relevant columns
cols <- c('PTS', 'ORB', 'DRB', 'W','X3P.','X2P.','FT.')

# Clean the data
req_cols <- df[,cols] 

req_cols
# Scale the data
df_scaled <- scale(req_cols)
rownames(df_scaled) <- df$Player



# Compute the distance matrix
df_dist <- dist(df_scaled, method = "cosine")


# Perform hierarchical clustering
df_hclust <- hclust(df_dist, method = "ward.D2")


# Plot the dendrogram
plot(df_hclust, cex = 0.6, main = "Hierarchical Clustering of top 50 NBA Players in 2018")
plot


fviz_dend(x = df_hclust, cex = 0.8, lwd = 0.8, k = 4,
          # Manually selected colors
          k_colors = c("jco"),
          rect = TRUE, 
          rect_border = "jco", 
          rect_fill = TRUE,
          main = "Hierarchical Clustering of top 50 NBA Players in 2018")
  

#install.packages("NbClust")
library(NbClust)
#install.packages('igraph')
library(igraph)



# Circular
Circ = fviz_dend(df_hclust, cex = 0.6, lwd = 0.6, k = 4,
                 rect = TRUE,
                 k_colors = "jco",
                 rect_border = "jco",
                 rect_fill = TRUE,
                 type = "circular")
Circ
-------------------


