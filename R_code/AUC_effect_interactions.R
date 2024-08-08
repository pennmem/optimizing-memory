library(lme4)
library(lmerTest)
library(emmeans)
library(tidyverse)
library(broom.mixed)

# read in properly labeled data
events <- read.csv("~/optimizing-memory/data/processed_events_with_AUC.csv")

# select only WORD trials with one of the three "stim" trial types
data <- events |>
  filter(type == "WORD", trial_type != "NoStim")

# group by subject and trial type and calculate the average recall
sub_recalled_score_df <- data |> group_by(subject, trial_type, AUC) |>
  summarize(sub_recalled = mean(recalled)) |>
  mutate(
    # code trial type as factor for negative, sham, positive
    trial_type = factor(trial_type, levels = c("Neg", "Sham", "Pos")),
    AUC_cent = AUC - .5 # center AUC at .5 (chance level prediction) for easier interpretation
  )

## fitting linear instead of categorical model, no correction for multiple comparisons needed.

sub_recalled_score_df$trial_type_num <- sub_recalled_score_df$trial_type %>% as.numeric() - 2
linear_model <- lmer(sub_recalled ~ trial_type_num * AUC_cent + (1 | subject), data=sub_recalled_score_df)
summary(linear_model)

broom_linear_model <- tidy(linear_model, conf.int=TRUE)
# write.csv(broom_linear_model, file = "~/optimizing-memory/results/linear_model.csv")

max_AUC <- max(sub_recalled_score_df$AUC_cent)
RG.max_AUC <- ref_grid(linear_model, at = list(AUC_cent = max_AUC), cov.keep = "trial_type_num" )
trial_type_num.emm <- emmeans(RG.max_AUC, specs=c("trial_type_num"))
plot(trial_type_num.emm)
plot(trial_type_num.emm, comparisons = TRUE)
pwpp(trial_type_num.emm)

broom_max_auc_emmeans <- tidy(trial_type_num.emm, conf.int=TRUE)
write.csv(broom_max_auc_emmeans, file = "~/optimizing-memory/results/max_auc_emmeans_linear.csv")
