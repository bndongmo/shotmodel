# Pre-scale the full dataset (excluding test set if necessary)
scaled_nba_shots <- scale_features(nba_shots, feature_cols)
scaled_shots_data <- scaled_nba_shots$data


results_list <- list()

for (id in unique(scaled_shots_data$player_id)) {
  player_shots <- scaled_shots_data %>%
    filter(player_id == id)
  
  pspp <- predict_shots_player_predictions(player_shots, id,
                                           prior_strength = 359.1743941748627, 
                                           multipliers = c(1.9335829032722032, 0.25322603744811045, 0.000305342832254224, 0.33397383545214837))
  player_shots$pred <- pspp$predictions
  player_shots$avg_similarity <- pspp$avg_similarities
  player_shots$rating <- player_shots$pred - player_shots$PRED_ZONE
  
  # Store the selected columns in a list
  results_list[[as.character(id)]] <- player_shots %>% select(SHOT_ID, pred, rating, avg_similarity)

}
# Bind all results together and join with nba_shots
final_results <- nba_shots %>%
  right_join(bind_rows(results_list), by = 'SHOT_ID')






