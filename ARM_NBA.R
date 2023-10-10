


df <- read.csv("test.csv")

df

library(viridis)
library(arules)
library(TSP)
#library(data.table)
#library(ggplot2)
#library(Matrix)
library(tcltk)
library(dplyr)
#install.packages('devtools')
library(devtools)
library(purrr)
library(tidyr)


trans <- read.transactions("test.csv",
                                rm.duplicates = FALSE, 
                                format = "basket",  ##if you use "single" also use cols=c(1,2)
                                sep=",",  ## csv file
                                cols=1) ## The dataset HAS row numbers
inspect(trans)



##### Use apriori to get the RULES
PrulesK = arules::apriori(trans, parameter = list(support=.25, 
                                                       confidence=.5, minlen=2))
inspect(PrulesK)

## Plot of which items are most frequent
itemFrequencyPlot(trans, topN=20, type="absolute")

## Sort rules by a measure such as conf, sup, or lift
SortedRules <- sort(PrulesK, by="support", decreasing=TRUE)
inspect(SortedRules[1:15])
(summary(SortedRules))

## Selecting or targeting specific rules  RHS
PtsRules <- apriori(data=trans,parameter = list(supp=.001, conf=.01, minlen=2),
                     appearance = list(default="lhs", rhs="20.1-30 Highest PTS"),
                     control=list(verbose=FALSE))
PtsRules <- sort(PtsRules, decreasing=TRUE, by="confidence")
inspect(PtsRules[1:10])

## Selecting rules with LHS specified
OREBRules <- apriori(data=trans,parameter = list(supp=.001, conf=.01, minlen=2),
                      appearance = list(default="rhs", lhs="3.0-4.0 ORB"),
                      control=list(verbose=FALSE))
OREBRules <- sort(OREBRules, decreasing=TRUE, by="support")
inspect(OREBRules[1:10])

#install.packages('arulesViz')
library(arulesViz)
## Visualize
## tcltk
#install.packages("arules")
library(arules)

plot(PrulesK, measure= 'lift', shading='confidence', method = "scatterplot", limit = 500, engine = "htmlwidget")

plot(PrulesK, measure= 'support', shading='lift', method = "scatterplot", limit = 500, engine = "htmlwidget")


plot(PrulesK, method = "graph", limit = 10)

plot(PrulesK, method = "graph", measure = "lift", shading = "confidence", engine = "htmlwidget")#, limit = 50)

plot(PrulesK, method = "matrix3D",engine = "htmlwidget")


plot(PrulesK, method = "graph", asEdges = TRUE, limit = 5, circular = FALSE)#, engine = "htmlwidget") 







