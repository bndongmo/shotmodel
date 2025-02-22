library(readr)
library(tidyverse)
library(glmnet)
library(parallel)
library(future.apply)

log_loss <- function(y_true, y_pred) {
  # y_true: True labels as a vector (e.g., 0 or 1 for binary classification)
  # y_pred: Predicted probabilities (values between 0 and 1 for binary classification)
  
  eps <- 1e-15  # To prevent log(0)
  y_pred <- pmin(pmax(y_pred, eps), 1 - eps)  # Clamp probabilities to avoid log(0)
  -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
}

transform_data <- function(new_data, lambdas) {
  # Apply transformations using the best lambdas
  transformed_data <- new_data %>%
    select(SHOT_ID, PERIOD, GAME_CLOCK, SHOT_NUMBER, SHOT_CLOCK, DRIBBLES, TOUCH_TIME, SHOT_DIST, CLOSE_DEF_DIST, FGM) %>%
    mutate(
      DRIBBLES = ((DRIBBLES + 1)^lambdas$DRIBBLES - 1) / lambdas$DRIBBLES,
      SHOT_DIST = ((SHOT_DIST + 1)^lambdas$SHOT_DIST - 1) / lambdas$SHOT_DIST,
      CLOSE_DEF_DIST = ((CLOSE_DEF_DIST + 1)^lambdas$CLOSE_DEF_DIST - 1) / lambdas$CLOSE_DEF_DIST,
      SHOT_CLOCK = ((SHOT_CLOCK + 1)^lambdas$SHOT_CLOCK - 1) / lambdas$SHOT_CLOCK,
      TOUCH_TIME = ((TOUCH_TIME + 1)^lambdas$TOUCH_TIME - 1) / lambdas$TOUCH_TIME,
      SHOT_NUMBER = ((SHOT_NUMBER + 1)^lambdas$SHOT_NUMBER - 1) / lambdas$SHOT_NUMBER
    )
  
  
  return(transformed_data)
}


# Define Box-Cox transformation function
boxcox_transform <- function(x, lambda) {
  if (abs(lambda) < 1e-4) {
    return(log(x))
  } else {
    return((x^lambda - 1) / lambda)
  }
}

log_loss <- function(y_true, y_pred) {
  # y_true: True labels as a vector (e.g., 0 or 1 for binary classification)
  # y_pred: Predicted probabilities (values between 0 and 1 for binary classification)
  
  eps <- 1e-15  # To prevent log(0)
  y_pred <- pmin(pmax(y_pred, eps), 1 - eps)  # Clamp probabilities to avoid log(0)
  -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
}




add_predictions <- function(shots) {
  num_shots <- nrow(shots)
  num_train_shots <- ceiling(num_shots/3)
  num_val_or_test_shots <- num_shots-ceiling(num_shots/3)
  
  
  train_shots <- shots %>% filter(set == 'train')
  val_shots <- shots %>% filter(set == 'val')
  test_shots <- shots %>% filter(set == 'test')
  
  
  # Define the loss function (negative AIC for minimization)
  
  loss_function <- function(lambdas) {
    lambdas <- lapply(lambdas, function(x) pmin(x, 200))
    names(lambdas) <- c("SHOT_NUMBER", "SHOT_CLOCK", "DRIBBLES", "TOUCH_TIME", "SHOT_DIST", "CLOSE_DEF_DIST")
    transformed_data <- transform_data(train_shots, as.list(lambdas))
    
    
    # Fit the logistic regression model
    model <- glm(FGM ~ SHOT_NUMBER + SHOT_CLOCK + DRIBBLES + TOUCH_TIME + SHOT_DIST + CLOSE_DEF_DIST, 
                 data = transformed_data, family = binomial)
    
    # Return the negative AIC
    return(AIC(model))
  }
  
  # Initial lambda values
  initial_lambdas <- c(
    SHOT_NUMBER = 1,
    SHOT_CLOCK = 1,
    DRIBBLES = 1,
    TOUCH_TIME = 1,
    SHOT_DIST = 1,
    CLOSE_DEF_DIST = 1
  )
  
  # Optimize 
  result <- optim(
    par = initial_lambdas,
    fn = loss_function,
    method = "Nelder-Mead",
  )
  
  # Extract optimal lambdas
  optimal_lambdas <- result$par
  names(optimal_lambdas) <- c("SHOT_NUMBER", "SHOT_CLOCK", "DRIBBLES", "TOUCH_TIME", "SHOT_DIST", "CLOSE_DEF_DIST")
  optimal_lambdas <- as.list(optimal_lambdas)
  
  # Print results
  cat("Optimal Lambda Values:\n")
  print(optimal_lambdas)
  
  train_data <- train_shots %>%
    transform_data(optimal_lambdas)
  
  val_data <- val_shots %>%
    transform_data(optimal_lambdas)
  
  train_x <- model.matrix(FGM ~ SHOT_NUMBER + SHOT_CLOCK + DRIBBLES + TOUCH_TIME + SHOT_DIST + CLOSE_DEF_DIST, data = train_data)[, -1]
  val_x <- model.matrix(FGM ~ SHOT_NUMBER + SHOT_CLOCK + DRIBBLES + TOUCH_TIME + SHOT_DIST + CLOSE_DEF_DIST, data = val_data)[, -1]
  
  train_y <- train_data$FGM
  val_y <- val_data$FGM
  
  # Define the parameter grids
  lambda_grid <- c(0, 10^seq(-3, 2, by = 0.05))
  alpha_grid <- seq(0, 1, by = 0.05)
  
  # Generate all combinations of alpha and lambda
  param_grid <- expand.grid(alpha = alpha_grid, lambda = lambda_grid)
  
  # Function to train and evaluate the model
  evaluate_model <- function(params) {
    alpha <- as.numeric(params["alpha"])
    lambda <- as.numeric(params["lambda"])
    
    model <- glmnet(train_x, train_y, alpha = alpha, lambda = lambda, family = "binomial")
    val_predictions <- predict(model, newx = val_x, type = "response")
    val_log_loss <- log_loss(val_y, val_predictions)
    
    return(c(alpha = alpha, lambda = lambda, log_loss = val_log_loss))
  }
  
  # Set up parallel cluster
  num_cores <- detectCores() - 1  # Leave one core free
  cl <- makeCluster(num_cores)
  
  # Export required variables and libraries to the workers
  clusterExport(cl, varlist = c("train_x", "train_y", "val_x", "val_y", "log_loss", "evaluate_model", "param_grid"))
  clusterEvalQ(cl, library(glmnet))
  
  # Perform parallel computation
  results <- parLapply(cl, seq_len(nrow(param_grid)), function(i) {
    evaluate_model(param_grid[i, ])
  })
  
  # Stop the cluster
  stopCluster(cl)
  
  # Convert results to a data frame
  results_df <- do.call(rbind, results)
  results_df <- as.data.frame(results_df)
  
  # Find the best parameters
  best_result <- results_df[which.min(results_df$log_loss), ]
  best_alpha <- as.numeric(best_result["alpha"])
  best_lambda <- as.numeric(best_result["lambda"])
  best_loss <- as.numeric(best_result["log_loss"])
  
  # Train the final model with best parameters
  best_model <- glmnet(train_x, train_y, alpha = best_alpha, lambda = best_lambda, family = "binomial")
  
  
  
  train_log_loss <- log_loss(train_y, predict(best_model, newx = train_x, type="response"))
  
  # Compute log losses
  train_log_loss_dummy <- log_loss(train_y, mean(train_y))
  val_log_loss_dummy <- log_loss(val_y, mean(train_y))
  
  cat("Train Log Loss", train_log_loss, "\n")
  cat("Val Log Loss", best_loss, "\n")
  
  cat("Train Log Loss (Dummy):", train_log_loss_dummy, "\n")
  cat("Val Log Loss (Dummy):", val_log_loss_dummy, "\n")
  
  cat("Improvement on Train Set:", train_log_loss_dummy - train_log_loss, "\n")
  cat("Improvement on Val Set:", val_log_loss_dummy - best_loss, "\n")
  
  all_data <- rbind(train_shots, val_shots, test_shots) %>%
    transform_data(optimal_lambdas)
  
  all_x <- model.matrix(FGM ~ SHOT_NUMBER + SHOT_CLOCK + DRIBBLES + TOUCH_TIME + SHOT_DIST + CLOSE_DEF_DIST, data = all_data)[, -1]
  
  
  shots$PRED_ZONE = predict(best_model, newx = all_x, type='response')
  
  return(list(shots = shots, model = best_model, lambdas = optimal_lambdas))
  
  
  
}
evaluate_split <- function(shots, var, thresh) {
  

  set1 <- shots %>% filter(!!sym(var) < thresh)
  set2 <- shots %>% filter(!!sym(var) >= thresh)
  
  if (nrow(set1 %>% filter(set == 'train')) < 2 | nrow(set2 %>% filter(set == 'train')) < 2) {
    return(list(results = NA, performance = Inf))
  }
  
  if (nrow(set1 %>% filter(set == 'val')) < 2 | nrow(set2 %>% filter(set == 'val')) < 2) {
    return(list(results = NA, performance = Inf))
  }

  set1_with_pred <- add_predictions(set1)
  set2_with_pred <- add_predictions(set2)

  
  
  results <- bind_rows(set1_with_pred, set2_with_pred) %>% na.omit()
  
  actual <- results %>% 
    filter(set == 'val') %>%
    mutate(made = if_else(SHOT_RESULT == 'made', 1, 0)) %>%
    pull(made)
  
  predicted <- results %>% 
    filter(set == 'val') %>%
    mutate(made = if_else(SHOT_RESULT == 'made', 1, 0)) %>%
    pull(PRED_ZONE)
  
  return(list(results = results, performance = log_loss(actual, predicted)))
}

# Logit and inverse logit functions
logit <- function(p) log(p / (1 - p))
inv_logit <- function(x) 1 / (1 + exp(-x))

# Function to compute similarity
feature_similarity <- function(vec1, vec2, multipliers) {
  res <- 1-abs(pnorm(vec1)-pnorm(vec2))
  return(prod(res^multipliers))
}

# Function to standardize features
scale_features <- function(data, feature_cols) {
  scaled_data <- scale(data[, feature_cols])
  
  # Store scaling parameters
  feature_means <- attr(scaled_data, "scaled:center")
  feature_sds <- attr(scaled_data, "scaled:scale")
  
  # Convert to dataframe
  scaled_df <- as.data.frame(scaled_data)
  scaled_df$FGM <- data$FGM
  scaled_df$PRED_ZONE <- data$PRED_ZONE
  scaled_df$player_id <- data$player_id
  scaled_df$player_name <- data$player_name
  scaled_df$set <- data$set
  scaled_df$SHOT_ID <- data$SHOT_ID
  
  return(list(data = scaled_df, means = feature_means, sds = feature_sds))
}



# Function to predict shots
predict_shots_player <- function(shots, id, prior_strength=100, multipliers = c(1,1,1,1)) {
  # Filter past shots for the relevant player
  past_shots_relevant <- shots %>%
    filter(player_id == id)
  
  # Initialize storage for results
  predictions = c(past_shots_relevant$PRED_ZONE[1])
  
  total_base_error = total_weighted_error = 0 
  
  # Iteratively predict each shot using previous shots
  for (i in 2:nrow(past_shots_relevant)) {
    
    # Use only past shots up to (i-1) for prediction
    historical_shots <- past_shots_relevant[1:(i-1), ]
    
    # Get the current shot to predict
    new_shot <- past_shots_relevant[i, feature_cols]
    
    # Compute similarity scores
    similarities <- apply(historical_shots[, feature_cols], 1, feature_similarity, vec2 = as.numeric(new_shot), multipliers = multipliers)
    
    # Compute weighted average FGM%
    if (sum(similarities) > 0) {
      weighted_FGM_percent <- sum(similarities * historical_shots$FGM) / sum(similarities)
    } else {
      weighted_FGM_percent <- mean(historical_shots$FGM)  # Fallback to average if no valid weights
    }
    
    # Compute weighted average expFGM%
    if (sum(similarities) > 0) {
      weighted_expFGM_percent <- sum(similarities * historical_shots$PRED_ZONE) / sum(similarities)
    } else {
      weighted_expFGM_percent <- mean(historical_shots$PRED_ZONE)  # Fallback to average if no valid weights
    }
    
    # Apply prior weighting
    weighted_FGM_percent <- (weighted_FGM_percent*(i-1) + weighted_expFGM_percent*prior_strength) / (prior_strength-1+i)
    
    # Get base model expected probability
    base_pred_prob <- past_shots_relevant[i, "PRED_ZONE"]  # Use predicted probability from base model
    
    # Compute log difference residual
    log_residual <- log(weighted_FGM_percent / (1 - weighted_FGM_percent)) - log(weighted_expFGM_percent / (1 - weighted_expFGM_percent))
    
    # Apply log difference to adjust the base model’s logit prediction
    logit_base_model_prediction <- logit(base_pred_prob)
    final_logit_prediction <- logit_base_model_prediction + log_residual
    
    # Convert back to probability scale
    final_prob_prediction <- inv_logit(final_logit_prediction)
    predictions = append(final_prob_prediction, predictions)
    
    # Evaluate improvement
    made = past_shots_relevant[i, "FGM"]
    base_error <- log_loss(made, base_pred_prob)  # Baseline log residual error
    weighted_error <- log_loss(made, final_prob_prediction)  # Error for this shot
    
    total_base_error = total_base_error + base_error
    total_weighted_error = total_weighted_error + weighted_error
  }
  return(total_weighted_error)
}


predict_shots <- function(shots, prior_strength=100, multipliers = c(1,1,1,1)) {
  plan(multisession, workers = parallel::detectCores() - 2)  # Use multiple cores
  
  # Run predictions in parallel for each player
  errors <- future_sapply(unique(shots$player_id), function(id) {
    predict_shots_player(as.data.frame(shots), id, prior_strength, multipliers)
  })
  
  plan(sequential)  # Reset parallelization to default
  
  
  return(sum(errors))  # Sum total error across all players
}


# Function to predict shots
predict_shots_player_predictions <- function(shots, id, prior_strength=100, multipliers = c(1,1,1,1)) {
  # Filter past shots for the relevant player
  past_shots_relevant <- shots %>%
    filter(player_id == id)
  
  # Initialize storage for results
  predictions = c(past_shots_relevant$PRED_ZONE[1])
  avg_similarities = c(.49)
  
  total_base_error = total_weighted_error = 0 
  
  # Iteratively predict each shot using previous shots
  for (i in 2:nrow(past_shots_relevant)) {
    
    # Use only past shots up to (i-1) for prediction
    historical_shots <- past_shots_relevant[1:(i-1), ]
    
    # Get the current shot to predict
    new_shot <- past_shots_relevant[i, feature_cols]
    
    # Compute similarity scores
    similarities <- apply(historical_shots[, feature_cols], 1, feature_similarity, vec2 = as.numeric(new_shot), multipliers = multipliers)
    avg_similarities <- c(avg_similarities, (sum(similarities)+4.9)/(9+i))
    
    # Compute weighted average FGM%
    if (sum(similarities) > 0) {
      weighted_FGM_percent <- sum(similarities * historical_shots$FGM) / sum(similarities)
    } else {
      weighted_FGM_percent <- mean(historical_shots$FGM)  # Fallback to average if no valid weights
    }
    
    # Compute weighted average expFGM%
    if (sum(similarities) > 0) {
      weighted_expFGM_percent <- sum(similarities * historical_shots$PRED_ZONE) / sum(similarities)
    } else {
      weighted_expFGM_percent <- mean(historical_shots$PRED_ZONE)  # Fallback to average if no valid weights
    }
    
    # Apply prior weighting
    weighted_FGM_percent <- (weighted_FGM_percent*(i-1) + weighted_expFGM_percent*prior_strength) / (prior_strength-1+i)
    
    # Get base model expected probability
    base_pred_prob <- past_shots_relevant[i, "PRED_ZONE"]  # Use predicted probability from base model
    
    # Compute log difference residual
    log_residual <- log(weighted_FGM_percent / (1 - weighted_FGM_percent)) - log(weighted_expFGM_percent / (1 - weighted_expFGM_percent))
    
    # Apply log difference to adjust the base model’s logit prediction
    logit_base_model_prediction <- logit(base_pred_prob)
    final_logit_prediction <- logit_base_model_prediction + log_residual
    
    # Convert back to probability scale
    final_prob_prediction <- inv_logit(final_logit_prediction)
    predictions = c(predictions, final_prob_prediction)
    
    
    # Evaluate improvement
    made = past_shots_relevant[i, "FGM"]
    base_error <- log_loss(made, base_pred_prob)  # Baseline log residual error
    weighted_error <- log_loss(made, final_prob_prediction)  # Error for this shot
    
    total_base_error = total_base_error + base_error
    total_weighted_error = total_weighted_error + weighted_error
    
  }
  
  return(list(predictions = predictions, avg_similarities = avg_similarities))
}


compare_shots <- function(shots, shot1, shot2, feature_cols, multipliers = c(1,1,1,1)) {
  # Ensure shots is a data frame
  shots <- as.data.frame(shots)
  
  # Get stored scaling parameters (means and sds)
  means <- attr(scale(shots[, feature_cols]), "scaled:center")
  sds <- attr(scale(shots[, feature_cols]), "scaled:scale")
  
  # Standardize hypothetical shots using dataset parameters
  shot1 <- (shot1 - means) / sds
  shot2 <- (shot2 - means) / sds
  
  # Compute similarity score
  similarity <- feature_similarity(shot1, shot2, multipliers)
  
  
  
  return(similarity)
}

predict_shot_for_player <- function(shots, id, new_shot, zone_models, feature_cols, prior_strength=100, multipliers = c(1,1,1,1)) {
  # Ensure shots is a data frame
  shots <- as.data.frame(shots)
  
  # Extract stored scaling parameters
  means <- attr(scale(shots[, feature_cols]), "scaled:center")
  sds <- attr(scale(shots[, feature_cols]), "scaled:scale")
  
  # Filter past shots for the relevant player
  past_shots_relevant <- shots %>%
    filter(player_id == id)
  
  if (nrow(past_shots_relevant) == 0) {
    stop("No past shots found for this player.")
  }
  
  # --- DETERMINE ZONE FOR NEW SHOT ---
  new_zone <- case_when(
    new_shot[1] <= 7 & new_shot[4] <= 1 & new_shot[1] <= 5 ~ "Restricted Area Quick Shot",
    new_shot[1] <= 7 & new_shot[4] <= 1 & new_shot[1] > 5 ~ "Paint Quick Shot",
    new_shot[1] <= 7 & new_shot[4] > 1 & new_shot[5] <= 2 ~ "Late Clock Interior Setup Shot",
    new_shot[1] <= 7 & new_shot[4] > 1 & new_shot[5] > 2 ~ "Interior Setup Shot",
    new_shot[1] > 7 & new_shot[2] <= 1 & new_shot[5] <= 3 ~ "Late Clock Closely Guarded Jumper",
    new_shot[1] > 7 & new_shot[2] <= 1 & new_shot[5] > 3 ~ "Closely Guarded Jumper",
    new_shot[1] > 7 & new_shot[2] > 1 & new_shot[4] <= 1 ~ "Catch-and-Shoot",
    new_shot[1] > 7 & new_shot[2] > 1 & new_shot[4] > 1 ~ "Setup Jumper",
    TRUE ~ "Unknown"
  )
  
  if (!(new_zone %in% names(zone_models))) {
    stop("No model found for zone: ", new_zone)
  }
  
  # --- RETRIEVE ZONE-SPECIFIC MODEL & LAMBDAS ---
  zone_info <- zone_models[[new_zone]]  
  base_model <- zone_info$model
  zone_lambdas <- zone_info$lambdas  
  
  # --- APPLY MANUAL TRANSFORMATIONS TO THE NEW SHOT ---
  new_shot_transformed <- data.frame(
    SHOT_NUMBER = ((new_shot[6] + 1)^zone_lambdas$SHOT_NUMBER - 1) / zone_lambdas$SHOT_NUMBER,
    SHOT_CLOCK = ((new_shot[5] + 1)^zone_lambdas$SHOT_CLOCK - 1) / zone_lambdas$SHOT_CLOCK,
    DRIBBLES = ((new_shot[3] + 1)^zone_lambdas$DRIBBLES - 1) / zone_lambdas$DRIBBLES,
    TOUCH_TIME = ((new_shot[4] + 1)^zone_lambdas$TOUCH_TIME - 1) / zone_lambdas$TOUCH_TIME,
    SHOT_DIST = ((new_shot[1] + 1)^zone_lambdas$SHOT_DIST - 1) / zone_lambdas$SHOT_DIST,
    CLOSE_DEF_DIST = ((new_shot[2] + 1)^zone_lambdas$CLOSE_DEF_DIST - 1) / zone_lambdas$CLOSE_DEF_DIST,
    FGM = 0
  )
  
  # --- USE `model.matrix()` TO ENSURE PROPER FORMAT FOR `PRED_ZONE` ---
  new_shot_matrix <- model.matrix(FGM ~ SHOT_NUMBER + SHOT_CLOCK + DRIBBLES + TOUCH_TIME + SHOT_DIST + CLOSE_DEF_DIST, data=new_shot_transformed)[, -1]
  
  # Compute base probability using the zone's logistic regression model
  base_pred_prob <- predict(base_model, newx = new_shot_matrix, type = "response")
  
  # --- STANDARDIZE FEATURES AFTER TRANSFORMATION ---
  past_shots_relevant[, feature_cols] <- sweep(past_shots_relevant[, feature_cols], 2, means, "-")
  past_shots_relevant[, feature_cols] <- sweep(past_shots_relevant[, feature_cols], 2, sds, "/")
  
  # Standardize the new shot for similarity computation
  new_shot_transformed <- (new_shot[1:4] - means) / sds
  
  # --- COMPUTE SIMILARITIES ---
  similarities <- apply(past_shots_relevant[, feature_cols], 1, feature_similarity, vec2 = as.numeric(new_shot_transformed), multipliers = multipliers)
  
  # Compute weighted average FGM%
  if (sum(similarities) > 0) {
    weighted_FGM_percent <- sum(similarities * past_shots_relevant$FGM) / sum(similarities)
  } else {
    weighted_FGM_percent <- mean(past_shots_relevant$FGM)  # Fallback if no valid weights
  }
  
  # Compute weighted average expFGM%
  if (sum(similarities) > 0) {
    weighted_expFGM_percent <- sum(similarities * past_shots_relevant$PRED_ZONE) / sum(similarities)
  } else {
    weighted_expFGM_percent <- mean(past_shots_relevant$PRED_ZONE)  # Fallback to average if no valid weights
  }
  
  
  # Apply prior weighting
  weighted_FGM_percent <- (weighted_FGM_percent * nrow(past_shots_relevant) + weighted_expFGM_percent * prior_strength) / (prior_strength + nrow(past_shots_relevant))
  
  # Compute final probability using logit adjustments
  log_residual <- log(weighted_FGM_percent / (1 - weighted_FGM_percent)) - log(weighted_expFGM_percent / (1 - weighted_expFGM_percent))
  avg_similarity = (sum(similarities)+4.9)/(10+length(similarities))
  similarity_residual <- boxcox_transform(avg_similarity, -1.8)*0.007372779
  final_logit_prediction <- logit(base_pred_prob) + log_residual + similarity_residual
  final_prob_prediction <- inv_logit(final_logit_prediction)
  return(list(base_pred = base_pred_prob, player_pred = final_prob_prediction, avg_similarity = avg_similarity))
}







# generate_player_predictions <- function(shots, a, prev_pred) {
#   data <- shots %>% 
#     mutate(res = FGM - !!sym(prev_pred),
#            FGA = 1
#     ) %>% 
#     group_by(player_id, player_name) %>%
#     mutate(total_shots = lag(cumsum(FGA), 1),
#            total_res = lag(cumsum(res), 1),
#            total_makes = lag(cumsum(FGM), 1),
#            pct = total_makes/total_shots,
#            exp_pct = (total_makes-total_res)/total_shots,
#            adj_pct = (total_makes+a*exp_pct)/(total_shots+a),
#            diff = log(adj_pct/(1-adj_pct))-log(exp_pct/(1-exp_pct)),
#            avg_res = total_res / total_shots,
#            offset = log(!!sym(prev_pred) / (1 - !!sym(prev_pred))),
#            offset = pmin(offset, 36)) %>%
#     replace_na(list(
#       total_shots = 0,
#       total_res = 0,
#       total_makes = 0,
#       pct = 0,
#       exp_pct = 0,
#       adj_pct = 0,
#       diff = 0,
#       avg_res = 0
#     )) %>%
#     mutate(player_pred = 1/(1+exp(-(offset+diff)))) %>%
#     select(player_id, player_name, set, SHOT_ID, SHOT_DIST, DRIBBLES, TOUCH_TIME, CLOSE_DEF_DIST, SHOT_NUMBER, SHOT_CLOCK, pct, exp_pct, total_shots, PRED_ZONE, player_pred, FGM)
#   
#   train <- data %>%
#     filter(set != 'test')
# 
#   data$rating = data$player_pred - data$PRED_ZONE
# 
# 
#   return(list(data = data, prev_perfomance = log_loss(train$FGM, train$PRED_ZONE), performance = log_loss(train$FGM, train$player_pred)))
#   
#   
# }
# 
# generate_player_predictions_test <- function(shots, bcl1, bcl2, prev_pred) {
#   data <- shots %>% 
#     mutate(res = if_else(SHOT_RESULT == 'made', 1, 0) - !!sym(prev_pred),
#            FGA = 1
#     ) %>% 
#     group_by(player_id, player_name) %>%
#     mutate(total_shots = lag(cumsum(FGA), 1),
#            total_res = lag(cumsum(res), 1),
#            avg_res = total_res / total_shots,
#            offset = log(!!sym(prev_pred) / (1 - !!sym(prev_pred))),
#            offset = pmin(offset, 36),
#            total_shots_bc = ((total_shots + 1)^bcl1 - 1) / bcl1,
#            avg_res_bc = ((avg_res + 1)^bcl2 - 1) / bcl2,
#            feature = total_shots_bc*avg_res_bc) %>%
#     replace_na(list(
#       total_shots = 0,
#       total_res = 0,
#       avg_res = 0,
#       total_shots_bc = 0,
#       avg_res_bc = 0,
#       feature = 0
#     ))
#   
#   train <- data %>%
#     filter(set != 'test')
#   
#   
#   glm_model <- glm(FGM ~ feature, family = binomial(link = "logit"), offset = offset, data = train)
#   data$player_pred <- predict(glm_model, newdata = data, type = "response", offset = data$offset)
#   data$rating = data$player_pred - data$PRED_ZONE
#   
#   val <- data %>%
#     filter(set == 'test')
#   
#   # Step 5: Model Summary
#   summary(glm_model) 
#   return(list(data = data, prev_perfomance = log_loss(val$FGM, val$PRED_ZONE), performance = log_loss(val$FGM, val$player_pred)))
#   
#   
# }
# 
# 
# optimize_prior <- function(shots, prev_pred) {
#   
#   # Objective function: Returns log loss for given bcl1 and bcl2
#   objective_function <- function(params) {
#     a <- params[1]
#   
#     # Run model with current bcl1 and bcl2 values
#     results <- generate_player_predictions(shots, a, prev_pred)
#     
#     # Extract validation log loss
#     log_loss <- results$performance
#     return(log_loss)  # We want to minimize this
#   }
#   
#   # Initial value
#   initial_params <- c(100)  
#   
#   # Optimize using Nelder-Mead
#   optim_result <- optim(
#     par = initial_params, 
#     fn = objective_function, 
#     method = "Nelder-Mead"
#   )
#   
#   return(optim_result)
# }
# 
# 
# 
# evaluate_player_split <- function(shots, var, thresh, prev_pred, min_train_size=30) {
#   
#   set1 <- shots %>% filter(!!sym(var) < thresh)
#   set2 <- shots %>% filter(!!sym(var) >= thresh)
#   
#   if (nrow(set1 %>% filter(set == 'train')) < min_train_size | nrow(set2 %>% filter(set == 'train')) < min_train_size) {
#     return(list(results = NA, performance = Inf))
#   }
#   
#   if (nrow(set1 %>% filter(set == 'val')) < min_train_size | nrow(set2 %>% filter(set == 'val')) < min_train_size) {
#     return(list(results = NA, performance = Inf))
#   }
#   
#   prior_1 = optimize_prior(set1, prev_pred)$par
#   prior_2 = optimize_prior(set2, prev_pred)$par
#   
#   results1 <- generate_player_predictions(set1, prior_1, prev_pred) 
#   results2 <- generate_player_predictions(set2, prior_2, prev_pred) 
#   
#   set1_with_pred <- results1$data
#   set2_with_pred <- results2$data
#   
#   
#   
#   results <- bind_rows(set1_with_pred, set2_with_pred) %>% na.omit()
#   
#   actual <- results %>% 
#     filter(set != 'test') %>%
#     pull(FGM)
#   
#   predicted <- results %>% 
#     filter(set != 'test') %>%
#     pull(player_pred)
#   
#   return(list(results = results, performance = log_loss(actual, predicted)))
#   
# }
# 
# 
# pick_split <- function(shots) {
#   best_var <- NA
#   best_overall <- Inf
#   best_split_value <- NA
#   best_results_overall <- NA
#   for (var in c("SHOT_DIST", "DRIBBLES", "TOUCH_TIME", "CLOSE_DEF_DIST")) {
#     best <- Inf
#     best_split <- NA
#     best_results <- NA
#     
#     
#     num = median(shots[[var]])
#     results <- try(evaluate_player_split(shots, var, num, 'player_pred', best_params), silent = TRUE)
#     
#     if (!inherits(results, "try-error")) {
#       if (!is.null(results)) {
#         performance <- results$performance
#         if (performance < 1) {
#           print(paste(var, num, performance))
#         }
#         
#         if (performance < best) {
#           best <- performance
#           best_split <- num
#           best_results <- results$results
#         }
#       }
#     }
#     
#     results <- try(evaluate_player_split(shots, var, num+0.1, 'player_pred', best_params), silent = TRUE)
#     
#     if (!inherits(results, "try-error")) {
#       if (!is.null(results)) {
#         performance <- results$performance
#         if (performance < 1) {
#           print(paste(var, num+.1, performance))
#         }
#         
#         if (performance < best) {
#           best <- performance
#           best_split <- num+.1
#           best_results <- results$results
#         }
#       }
#     }
#     
#     
#     if (best < best_overall) {
#       best_overall <- best
#       best_var <- var
#       best_split_value <- best_split
#       best_results_overall <- best_results
#     }
#   }
#   
#   cat("Best variable:", best_var, "\nBest split value:", best_split_value, "\nBest performance:", best_overall)
#   set1 <- shots %>% filter(!!sym(best_var) < best_split_value)
#   set2 <- shots %>% filter(!!sym(best_var) >= best_split_value)
#   return(list(set1, set2))
# }
# 
# pick_split_recursive <- function(shots, depth) {
#   if (depth == 0 || nrow(shots) < 2) {
#     # Base case: stop recursion when depth is 0 or if the set is too small
#     return(list(shots))
#   }
#   
#   best_var <- NA
#   best_overall <- Inf
#   best_split_value <- NA
#   best_results_overall <- NA
#   
#   for (var in c("SHOT_DIST", "DRIBBLES", "TOUCH_TIME", "CLOSE_DEF_DIST")) {
#     best <- Inf
#     best_split <- NA
#     best_results <- NA
#     
#     num <- median(shots[[var]], na.rm = TRUE)  # Median split to prevent extreme values
#     results <- try(evaluate_player_split(shots, var, num, 'PRED_ZONE'), silent = TRUE)
#     
#     if (!inherits(results, "try-error") && !is.null(results)) {
#       performance <- results$performance
#       if (performance < 1) {
#         print(paste(var, num, performance))
#       }
#       
#       if (performance < best) {
#         best <- performance
#         best_split <- num
#         best_results <- results$results
#       }
#     }
#     
#     # Try a slight adjustment in case of ties or local optima
#     results <- try(evaluate_player_split(shots, var, num + 1e-8, 'PRED_ZONE'), silent = TRUE)
#     
#     if (!inherits(results, "try-error") && !is.null(results)) {
#       performance <- results$performance
#       if (performance < 1) {
#         print(paste(var, num + 1e-8, performance))
#       }
#       
#       if (performance < best) {
#         best <- performance
#         best_split <- num + 1e-8
#         best_results <- results$results
#       }
#     }
#     
#     # Update best overall split
#     if (best < best_overall) {
#       best_overall <- best
#       best_var <- var
#       best_split_value <- best_split
#       best_results_overall <- best_results
#     }
#   }
#   
#   # If no valid split is found, return the original set
#   if (is.na(best_var)) {
#     return(list(shots))
#   }
#   
#   cat("Best variable:", best_var, "\nBest split value:", best_split_value, "\nBest performance:", best_overall, "\n")
#   
#   # Split dataset
#   set1 <- best_results_overall %>% filter(!!sym(best_var) < best_split_value)
#   set2 <- best_results_overall %>% filter(!!sym(best_var) >= best_split_value)
#   
#   # Recursively split both subsets until depth is reached
#   split1 <- pick_split_recursive(set1, depth - 1)
#   split2 <- pick_split_recursive(set2, depth - 1)
#   
#   # Combine all leaf nodes
#   return(c(split1, split2))
# }
# 
