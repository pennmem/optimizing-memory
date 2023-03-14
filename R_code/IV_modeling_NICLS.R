library(tidyverse)
library(lme4)
library(lmerTest)
library(broom.mixed)

df <- read.csv("Downloads/processed_events_NiclsCourierClosedLoop.csv")

df <- df |> filter(type == "WORD", trial_type != "NoStim")
levels(df$trial_type) <- c("Sham", "Neg", "Pos")
df$trial_type <- factor(df$trial_type, levels = c("Sham", "Neg", "Pos"))


# Positive vs. Negative ---------------------------------------------------

pos_neg_df <- df |> filter(trial_type != "Sham")

first_stage_model <- lmer(probability ~ trial_type + (trial_type | subject/session), data=pos_neg_df)
summary(first_stage_model)

second_stage_model <- lmer(recalled ~ trial_type + (trial_type | subject/session), data=pos_neg_df)
summary(second_stage_model)

second_stage_model2 <- lmer(recalled ~ trial_type + (dummy(trial_type) || subject) + (trial_type | subject:session), data=pos_neg_df)
summary(second_stage_model2)

second_stage_model3 <- lmer(recalled ~ trial_type + (1 | subject) + (trial_type | subject:session), data=pos_neg_df)
summary(second_stage_model3)

# Positive vs. Sham -------------------------------------------------------

pos_sham_df <- df |> filter(trial_type != "Neg")
pos_sham_df$trial_type <- factor(pos_sham_df$trial_type, levels = c("Sham", "Pos"))

first_stage_model <- lmer(probability ~ trial_type + (trial_type | subject/session), data=pos_sham_df)
summary(first_stage_model)

second_stage_model <- lmer(recalled ~ trial_type + (trial_type | subject/session), data=pos_sham_df)
summary(second_stage_model)

second_stage_model2 <- lmer(recalled ~ trial_type + (dummy(trial_type) || subject) + (trial_type | subject:session), data=pos_sham_df)
summary(second_stage_model2)

second_stage_model3 <- lmer(recalled ~ trial_type + (1 | subject) + (trial_type | subject:session), data=pos_sham_df)
summary(second_stage_model3)


# Negative vs. Sham -------------------------------------------------------

neg_sham_df <- df |> filter(trial_type != "Pos")
neg_sham_df$trial_type <- factor(neg_sham_df$trial_type, levels = c("Sham", "Neg"))

first_stage_model <- lmer(probability ~ trial_type + (trial_type | subject/session), data=neg_sham_df)
summary(first_stage_model)

second_stage_model <- lmer(recalled ~ trial_type + (trial_type | subject/session), data=neg_sham_df)
summary(second_stage_model)

second_stage_model2 <- lmer(recalled ~ trial_type + (trial_type | subject) + (1 | subject:session), data=neg_sham_df)
summary(second_stage_model2)


# Group mean models -------------------------------------------------------

sess_df <- df |> 
  group_by(subject, session, trial_type) |> 
  summarize(mean_sess_recalled = mean(recalled, na.rm = TRUE), mean_sess_probability = mean(probability, na.rm = TRUE))

sub_df <- sess_df |>
  group_by(subject, trial_type) |> 
  summarize(mean_sub_recalled = mean(mean_sess_recalled, na.rm = TRUE), mean_sub_probability = mean(mean_sess_probability, na.rm = TRUE))

ggplot(sub_df, aes(mean_sub_recalled, mean_sub_probability)) + geom_point() + stat_smooth()

m1 <- lmer(mean_sub_recalled ~ mean_sub_probability + (1 | subject), data = sub_df)
summary(m1)

m2 <- lmer(mean_sess_recalled ~ mean_sess_probability + (mean_sess_probability | subject), data = sess_df)
summary(m2)

m2a <- lmer(mean_sess_recalled ~ mean_sess_probability + (mean_sess_probability || subject), data = sess_df)
summary(m2a)

m2b <- lmer(mean_sess_recalled ~ mean_sess_probability + (1 | subject), data = sess_df)
summary(m2b)
