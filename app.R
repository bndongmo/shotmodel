library(shiny)
library(dplyr)
library(DT)
library(shinyWidgets)

# Load necessary functions
source("functions.R")

# Load pre-saved datasets
nba_shots <- readRDS("nba_shots_with_base_predictions.rds")
zone_models <- readRDS("zone_models.rds")
feature_cols <- readRDS("feature_cols.rds")

# Define UI
ui <- fluidPage(
  titlePanel("Basketball Shot Similarity & Prediction App"),
  
  tabsetPanel(
    # Tab 1: Compare Shots
    tabPanel("Compare Shots",
             sidebarLayout(
               sidebarPanel(
                 h3("Shot 1"),
                 numericInput("shot1_dist", "SHOT_DIST:", value = 25, min = 0, max = 300),
                 numericInput("shot1_def", "CLOSE_DEF_DIST:", value = 3, min = 0, max = 100),
                 numericInput("shot1_drib", "DRIBBLES:", value = 2, min = 0, max = 100),
                 numericInput("shot1_touch", "TOUCH_TIME:", value = 0, min = 0, max = 100),
                 
                 h3("Shot 2"),
                 numericInput("shot2_dist", "SHOT_DIST:", value = 25, min = 0, max = 300),
                 numericInput("shot2_def", "CLOSE_DEF_DIST:", value = 3, min = 0, max = 100),
                 numericInput("shot2_drib", "DRIBBLES:", value = 2, min = 0, max = 100),
                 numericInput("shot2_touch", "TOUCH_TIME:", value = 1.5, min = 0, max = 100),
                 
                 actionButton("compare_button", "Compare Shots")
               ),
               mainPanel(
                 h3("Shot Similarity Score"),
                 verbatimTextOutput("similarity_output")
               )
             )
    ),
    
    # Tab 2: Predict Shots for Selected Players
    tabPanel("Predict Shot for Players",
             sidebarLayout(
               sidebarPanel(
                 h3("Hypothetical Shot"),
                 numericInput("pred_shot_dist", "SHOT_DIST:", value = 27, min = 0, max = 300),
                 numericInput("pred_shot_def", "CLOSE_DEF_DIST:", value = 10, min = 0, max = 100),
                 numericInput("pred_shot_drib", "DRIBBLES:", value = 3, min = 0, max = 100),
                 numericInput("pred_shot_touch", "TOUCH_TIME:", value = 2.5, min = 0, max = 100),
                 numericInput("pred_shot_clock", "SHOT_CLOCK:", value = 5, min = 0, max = 24),
                 numericInput("pred_shot_number", "SHOT_NUMBER:", value = 10, min = 0, max = 500),
                 
                 h3("Select Players"),
                 actionButton("select_all", "Select All"),
                 actionButton("deselect_all", "Deselect All"),
                 
                 pickerInput("selected_players", "Choose Players:", 
                             choices = unique(nba_shots$player_name), 
                             selected = unique(nba_shots$player_name)[1:10], 
                             multiple = TRUE,
                             options = list(`actions-box` = TRUE, `live-search` = TRUE)),
                 
                 actionButton("predict_button", "Predict for Selected Players")
               ),
               mainPanel(
                 h3("Base Model Prediction"),
                 verbatimTextOutput("base_pred_output"),
                 h3("Prediction Results"),
                 DTOutput("predictions_table")
               )
             )
    )
  )
)

# Define Server
server <- function(input, output, session) {
  
  # Compare Shots
  observeEvent(input$compare_button, {
    shot1 <- c(input$shot1_dist, input$shot1_def, input$shot1_drib, input$shot1_touch)
    shot2 <- c(input$shot2_dist, input$shot2_def, input$shot2_drib, input$shot2_touch)
    
    similarity <- compare_shots(nba_shots, shot1, shot2, feature_cols, 
                                multipliers = c(1.9335829032722032, 0.25322603744811045, 0.000305342832254224, 0.33397383545214837))
    
    output$similarity_output <- renderText({
      paste("Shot Similarity Score:", round(similarity, 4))
    })
  })
  
  # Select All Players
  observeEvent(input$select_all, {
    updatePickerInput(session, "selected_players",
                      selected = unique(nba_shots$player_name))
  })
  
  # Deselect All Players
  observeEvent(input$deselect_all, {
    updatePickerInput(session, "selected_players",
                      selected = character(0))  # Empty selection
  })
  
  # Predict Shots for Selected Players
  observeEvent(input$predict_button, {
    new_shot <- c(input$pred_shot_dist, input$pred_shot_def, input$pred_shot_drib, 
                  input$pred_shot_touch, input$pred_shot_clock, input$pred_shot_number)
    
    # Get the base model probability (same for all players)
    base_pred <- predict_shot_for_player(nba_shots, nba_shots$player_id[1], new_shot, zone_models, feature_cols, 
                                         prior_strength = 359.1743941748627, 
                                         multipliers = c(1.9335829032722032, 0.25322603744811045, 0.000305342832254224, 0.33397383545214837))$base_pred
    
    # Display the base prediction separately
    output$base_pred_output <- renderText({
      paste("Base Model Prediction for This Shot:", round(base_pred, 4))
    })
    
    # Filter players based on user selection
    selected_players <- nba_shots %>% filter(player_name %in% input$selected_players) %>% pull(player_id) %>% unique()
    
    predictions <- lapply(selected_players, function(id) {
      player_name <- nba_shots %>% filter(player_id == id) %>% pull(player_name) %>% unique()
      
      result <- predict_shot_for_player(nba_shots, id, new_shot, zone_models, feature_cols, 
                                        prior_strength = 359.1743941748627, 
                                        multipliers = c(1.9335829032722032, 0.25322603744811045, 0.000305342832254224, 0.33397383545214837))
      
      data.frame(Player_ID = id, Player_Name = player_name, Player_Pred = round(as.numeric(result$player_pred), 3), Avg_Similarity = round(result$avg_similarity, 3))
    })
    
    # Combine into a single dataframe
    predictions_df <- do.call(rbind, predictions)
    
    # Render in DataTable
    output$predictions_table <- renderDT({
      datatable(predictions_df, options = list(pageLength = 10))
    })
  })
}

shinyApp(ui = ui, server = server)

