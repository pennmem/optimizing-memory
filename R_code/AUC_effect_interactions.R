library(lme4)
library(lmerTest)
library(emmeans)
library(tidyverse)
library(broom.mixed)

events <- read.csv("~/optimizing-memory/data/processed_events_with_AUC.csv")
data <- events |>
  filter(type == "WORD", trial_type != "NoStim")

sub_recalled_score_df <- data |> group_by(subject, trial_type, AUC) |>
  summarize(sub_recalled = mean(recalled)) |>
  mutate(
    trial_type = factor(trial_type, levels = c("Neg", "Sham", "Pos")),
    AUC_cent = AUC - .5
  )
three_way_model <- lmer(sub_recalled ~ trial_type * AUC_cent + (1 | subject), data=sub_recalled_score_df)
summary(three_way_model)

broom_three_way_model <- tidy(three_way_model, conf.int=TRUE)
write.csv(broom_three_way_model, file = "~/optimizing-memory/results/three_way_model_neg_intercept.csv")

ggplot(broom_three_way_model, aes(x=reorder(term, estimate), y=estimate)) +
  geom_errorbar(aes(ymin=conf.low, ymax=conf.high), 
                width = 0.2,size  = 1,
                position = "dodge", color="turquoise4") +
  geom_hline(yintercept = 0, color = "red", size = 1) +
  geom_point() + coord_flip()


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
plot(trial_type.emm)
plot(trial_type.emm, comparisons = TRUE)
pwpp(trial_type.emm)

emmeans(RG.max_AUC, ~ trial_type)


## fitting linear instead of categorical model,  no correction for multiple comparisons needed.

sub_recalled_score_df$trial_type_num <- sub_recalled_score_df$trial_type %>% as.numeric() - 2
linear_model <- lmer(sub_recalled ~ trial_type_num * AUC_cent + (1 | subject), data=sub_recalled_score_df)
summary(linear_model)

broom_linear_model <- tidy(linear_model, conf.int=TRUE)
write.csv(broom_linear_model, file = "~/optimizing-memory/results/linear_model.csv")
