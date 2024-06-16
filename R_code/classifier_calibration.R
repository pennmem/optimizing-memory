library(lme4)
library(lmerTest)
library(emmeans)
library(tidyverse)
library(broom.mixed)

# read in labels / predictions
data <- read.csv("~/optimizing-memory/data/RO_encoding_loso_calibration_data.csv")

# create column for predicted label
data$predicted_label <- ifelse(data$predicted > 0.5, 1, 0)

filtered_data <- data |> 
  filter(predicted_label == 1)
    
# fit a mixed effects model with random slopes and intercepts for session nested 
# within subject
model <- lmer(recalled ~ predicted + (predicted|subject/session), data = filtered_data)
summary(model)

data |> group_by(subject) |> 
  summarize(
    mean_recalled = mean(recalled),
    mean_predicted = mean(predicted_label),
    n = n()
  )

# bin by predicted within subject
subject_bin_means <- data |> 
  group_by(subject) |> 
  mutate(
    predicted_ntile = ntile(predicted, 10),
    
  ) |>
  group_by(subject, predicted_ntile) |>
  summarize(
    mean_recalled = mean(recalled),
    mean_predicted = mean(predicted)
  )
subject_bin_means

# for each subject plot recalled by predicted
subject_bin_means |> 
  ggplot(aes(x=mean_predicted, y=mean_recalled)) +
  geom_point() +
  geom_smooth(method = "lm") +
  facet_wrap(~subject)

                                       