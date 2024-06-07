library(lme4)
library(lmerTest)
library(emmeans)
library(tidyverse)
library(broom.mixed)

# read in labels / predictions
data <- read.csv("~/optimizing-memory/data/RO_encoding_loso_calibration_data.csv")

# create column for predicted label
data$predicted_label <- ifelse(data$predicted > 0.5, 1, 0)
    
# fit a mixed effects model with random slopes and intercepts for session nested 
# within subject
model <- lmer(recalled ~ predicted + (predicted|subject/session), data = data)
summary(model)

data |> group_by(subject) |> 
  summarize(
    mean_recalled = mean(recalled),
    mean_predicted = mean(predicted_label),
    n = n()
  )

                                       