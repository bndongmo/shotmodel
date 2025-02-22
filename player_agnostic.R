best_var <- NA
best_overall <- Inf
best_split_value <- NA
best_results_overall <- NA

for (var in c("SHOT_NUMBER", "SHOT_DIST", "SHOT_CLOCK", "DRIBBLES", "TOUCH_TIME", "CLOSE_DEF_DIST")) {
  best <- Inf
  best_split <- NA
  best_results <- NA
  
  for (num in 1:100) {
    results <- evaluate_split(nba_shots %>% filter(SHOT_DIST >= 7, CLOSE_DEF_DIST >= 1), var, num)
    
    if (!is.null(results)) {
      performance <- results$performance
      
      if (performance < best) {
        best <- performance
        best_split <- num
        best_results <- results$results
      } else {
        break
      }
    }
  }
  
  if (best < best_overall) {
    best_overall <- best
    best_var <- var
    best_split_value <- best_split
    best_results_overall <- best_results
  }
}

cat("Best variable:", best_var, "\nBest split value:", best_split_value, "\nBest performance:", best_overall)