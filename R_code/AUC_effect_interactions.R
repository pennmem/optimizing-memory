library(lme4)
library(lmerTest)
library(emmeans)
library(tidyverse)
library(broom.mixed)

df <- read.csv("~/optimizing-memory/data/processed_events_with_AUC.csv")
data <- df |>
  filter(type == "WORD", trial_type != "NoStim")

sub_recalled_score_df <- data %>% group_by(subject, trial_type, AUC) |>
  summarize(sub_recalled = mean(recalled)) |>
  mutate(
    trial_type = factor(trial_type, levels = c("Neg", "Sham", "Pos")),
    AUC_cent = AUC - .5
  )
three_way_model <- lmer(sub_recalled ~ trial_type * AUC_cent + (1 | subject), data=sub_recalled_score_df)
summary(three_way_model)

broom_three_way_model <- tidy(three_way_model)
write.csv(broom_three_way_model, file = "~/optimizing-memory/results/three_way_model_neg_intercept.csv")

#ANOVA for interaction terms
anova(three_way_model)

#post-hoc test of pairwise differences in slopes
auc.emt <- emtrends(three_way_model, pairwise ~ trial_type, var = "AUC_cent")
#no p-value adjustment
summary(auc.emt, adjust = "none")
#tukey adjustment (probably correct)
summary(auc.emt)

#post-hoc test of pairwise differences in trial-type at highest AUC
max_AUC <- max(sub_recalled_score_df$AUC_cent)
RG.max_AUC <- ref_grid(three_way_model, at = list(AUC_cent = max_AUC))
trial_type.emm <- emmeans(RG.max_AUC, pairwise ~ trial_type)
#no p-value adjustment
summary(trial_type.emm, adjust = "none")
#tukey adjustment (probably correct)
summary(trial_type.emm)

