# Search for optimal lambda manually
lambda_seq <- seq(-4, 0, by = 0.1)  # Range of lambda values to test
aic_values <- numeric(length(lambda_seq))

for (i in seq_along(lambda_seq)) {
  lambda <- lambda_seq[i]

  # Split data into train and validation sets
  train_data <- final_results
  
  # Ensure avg_similarity is transformed
  train_data$avg_similarity_trans <- boxcox_transform(train_data$avg_similarity, lambda)
  
  X_train <- model.matrix(FGM ~ avg_similarity_trans, data = train_data)
  y_train <- train_data$FGM
  offset_train <- pmin(logit(train_data$pred), 36)
  
  # Fit regularized logistic regression (Ridge with alpha=0)
  model <- glmnet(X_train, y_train, alpha = 0, lambda = 0, family = "binomial", offset = offset_train, intercept=FALSE)
  
  # Predict on validation set
  sim_pred <- predict(model, newx = X_train, newoffset = offset_train, type = "response")
  
  aic_values[i] <- mean(log_loss(y_train, sim_pred))
}

# Find the lambda that minimizes AIC
optimal_lambda <- lambda_seq[which.min(aic_values)]
cat("Optimal Lambda:", optimal_lambda, "\n")








# Split data into train and validation sets
train_data <- final_results %>% filter(set == "train")
val_data <- final_results %>% filter(set == "val")

# Ensure avg_similarity is transformed
train_data$avg_similarity_trans <- boxcox_transform(train_data$avg_similarity, optimal_lambda)
val_data$avg_similarity_trans <- boxcox_transform(val_data$avg_similarity, optimal_lambda)

X_train <- model.matrix(FGM ~ avg_similarity_trans, data = train_data)
X_val <- model.matrix(FGM ~ avg_similarity_trans, data = val_data)

y_train <- train_data$FGM
y_val <- val_data$FGM

offset_train <- pmin(logit(train_data$pred), 36)
offset_val <- pmin(logit(val_data$pred), 36)

lambda_seq2 <- c(0, 10^seq(-4, 4, length.out = 50))  # Log scale search space
cv_results <- data.frame(lambda = lambda_seq2, val_log_loss = NA, train_log_loss = NA)

for (j in seq_along(lambda_seq2)) {
  lambda2 <- lambda_seq2[j]
  
  # Fit regularized logistic regression (Ridge with alpha=0)
  model <- glmnet(X_train, y_train, alpha = 0, lambda = lambda2, family = "binomial", offset = offset_train, intercept=FALSE)
  
  # Predict on validation set
  sim_pred <- predict(model, newx = X_val, newoffset = offset_val, type = "response")
  train_sim_pred <- predict(model, newx = X_train, newoffset = offset_train, type = "response")
  
  # Compute validation log loss
  cv_results$val_log_loss[j] <- log_loss(y_val, sim_pred)
  cv_results$train_log_loss[j] <- log_loss(y_train, train_sim_pred)
}

# Find best lambda
best_lambda <- cv_results$lambda[which.min(cv_results$val_log_loss)]
cat("Optimal Regularization Lambda:", best_lambda, "\n")




# Split data into train and validation sets
train_data <- final_results %>% filter(set == "train")
val_data <- final_results %>% filter(set == "val")


# Ensure avg_similarity is transformed
train_data$avg_similarity_trans <- boxcox_transform(train_data$avg_similarity, optimal_lambda)
val_data$avg_similarity_trans <- boxcox_transform(val_data$avg_similarity, optimal_lambda)

# Prepare features and response for glmnet
X_train <- model.matrix(FGM ~ avg_similarity_trans, data = train_data)
X_val <- model.matrix(FGM ~ avg_similarity_trans, data = val_data)

y_train <- train_data$FGM
y_val <- val_data$FGM

offset_train <- pmin(logit(train_data$pred), 36)
offset_val <- pmin(logit(val_data$pred), 36)



# Fit final model with best lambda
final_model <- glmnet(X_train, y_train, alpha = 0, lambda = 0, family = "binomial", offset = offset_train, intercept = FALSE)
coef(final_model)
# Predict using final model
final_results$avg_similarity_trans <- boxcox_transform(final_results$avg_similarity, optimal_lambda)
final_results$sim_pred <- predict(final_model, newx = model.matrix(FGM ~ avg_similarity_trans, data = final_results),
                                  newoffset = pmin(logit(final_results$pred), 36), type = "response")
final_results$sim_rating <- final_results$sim_pred - final_results$PRED_ZONE

# Evaluate errors
final_results %>%
  group_by(set) %>%
  reframe(base_error = mean(log_loss(FGM, PRED_ZONE)), 
          player_error = mean(log_loss(FGM, pred)), 
          sim_error = mean(log_loss(FGM, sim_pred))) %>%
  View()



