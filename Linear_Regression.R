# Load necessary libraries
library(readr)
library(dplyr)
library(caret)
library(corrplot)

# Load the dataset
data <- read_csv("/Users/shaikjunaid/Downloads/Cleaned_Reg.csv")


# calculate the correlation matrix
corr <- cor(data)
# print the correlation matrix
print(corr)
# plot the correlation matrix
corrplot(corr)


# Display the first few rows (Before Preprocessing)
print(head(data))

# Preprocessing steps
#Select relevant features
data_preprocessed <- data %>%
  select(Year, PTS, GP, MIN, OREB, DREB, FG_PCT, FT_PCT, AST, Avg_Points)

#Convert 'Year' to a numeric format for splitting
data_preprocessed$Year <- as.numeric(sub("-.*", "", data_preprocessed$Year))

#Handle missing values (example: replace with median or mean)
median_values <- data_preprocessed %>%
  summarize(across(where(is.numeric), median, na.rm = TRUE))

data_preprocessed <- data_preprocessed %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.), median_values[[cur_column()]], .)))

# Display the first few rows (After Preprocessing)
print(head(data_preprocessed))


# Splitting the data
train <- filter(data, Year >= 2012 & Year <= 2018)
test <- filter(data, Year >= 2019 & Year <= 2021)

write.csv(train, "train.csv", row.names = FALSE)
write.csv(test, "test.csv", row.names = FALSE)
head(train)
head(test)
-----------------------------------
# Model Development - Using Linear Regression
model <- lm(Avg_Points ~ ., data = test)

# Prediction and Evaluation
predictions <- predict(model, test)

# Calculate Mean Squared Error
mse <- mean((predictions - test$Avg_Points)^2)

# Output the MSE and model summary
print(paste("Mean Squared Error: ", mse))
summary(model)

-----------------------------
  
library(ggplot2)

# Add the predictions to the test set for plotting
test$Predicted <- predictions

# Create the plot
ggplot(test, aes(x = Avg_Points, y = Predicted)) +
  geom_point() +
  geom_abline(color = "red", linetype = "dashed") +
  ggtitle("Actual vs Predicted Average Points") +
  xlab("Actual Average Points") +
  ylab("Predicted Average Points") +
  theme_minimal()


-------------------------------
  
plot(model)


---------------------------------
  
  # Model Development - Using Linear Regression
model1 <- lm(Avg_Points ~ `PTS`, data = test)

# Prediction and Evaluation
predictions <- predict(model1, test)

# Calculate Mean Squared Error
mse <- mean((predictions - test$Avg_Points)^2)

# Output the MSE and model summary
print(paste("Mean Squared Error: ", mse))
summary(model1)

----------------------------------
  
# Compare model 1 to model 2
anova(model, model1)