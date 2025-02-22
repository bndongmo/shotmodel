# Player-Adjusted Shot Prediction Model

## Overview

This project implements a **player-adjusted shot prediction model** that improves upon a standard player-agnostic model by incorporating **historical player tendencies**. The model accounts for **shot similarity weighting** and **penalizes out-of-profile shots**, ensuring more realistic predictions.

The model is integrated into an **RShiny app**, which can be accessed at:

🔗 [**RShiny App Link**](https://bndongmo.shinyapps.io/shot_model/)

## Features

- **Player-Specific Adjustments:** Uses historical player data to modify shot probability predictions.
- **Similarity-Based Weighting:** Prior shots influence predictions based on how similar they are to new attempts.
- **Penalizing Out-of-Profile Shots:** Reduces overestimation of players attempting shots outside their usual profile.
- **Hyperparameter Optimization:** Uses **Optuna** for optimizing feature weights and prior strength.
- **Regularized Logistic Regression:** Final predictions are computed via **ridge regression** to prevent overfitting.
- **Integration with RShiny:** Users can interact with the model through a dynamic web app.

## Installation

### Requirements

Ensure you have the following dependencies installed:

- R (>= 4.0.0)
- RShiny
- Tidyverse
- glmnet
- reticulate
- future.apply
- parallel
- Python (for Optuna optimization)

### Installation Steps

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/player-shot-model.git
   cd player-shot-model
   ```
2. Install R dependencies:
   ```r
   install.packages(c("shiny", "tidyverse", "glmnet", "reticulate", "future.apply", "parallel"))
   ```
3. Install Python dependencies:
   ```sh
   pip install optuna numpy
   ```

## Usage

### Using the RShiny App

Visit the [**RShiny App**](https://bndongmo.shinyapps.io/shot_model/) to interact with the model.

## Methodology

### Player-Agnostic Model

Before incorporating player-specific adjustments, we first develop a **player-agnostic model** that predicts shot success probability using only shot context features. This model does not account for individual player tendencies but instead relies on general shot characteristics, including:
- **Shot Distance (`SHOT_DIST`)** – The farther the shot, the lower the probability of success.
- **Defender Proximity (`CLOSE_DEF_DIST`)** – A closer defender decreases shot probability.
- **Dribbles (`DRIBBLES`)** – Catch-and-shoot attempts often have higher success rates than off-the-dribble shots.
- **Touch Time (`TOUCH_TIME`)** – Longer possession times can indicate more difficult attempts.
- **Shot Clock (`SHOT_CLOCK`)** – Shorter shot clock situations may lead to rushed or lower-quality shots.
- **Shot Number (`SHOT_NUMBER`)** – Later shots in a player's sequence may be influenced by fatigue or game adjustments.

A **logistic regression model** is trained using these features, estimating the likelihood of a shot being made (`FGM`). This model serves as the **baseline prediction (`PRED_ZONE`)** before player-specific adjustments are applied.

However, **SHOT_CLOCK** and **SHOT_NUMBER** are excluded from player-specific similarity adjustments, as specialization effects for these features are believed to be minimal. Instead, the similarity-based approach focuses on core shot features that better reflect player tendencies.

To improve the accuracy of this model, we introduce **shot zoning**, where shots are classified into different categories based on their characteristics. By separating shots into meaningful **zones**, we account for differences in shot difficulty across different types of attempts. The dataset is divided into the following zones:
- **Restricted Area Quick Shot** – Close-range shots with minimal touch time.
- **Paint Quick Shot** – Slightly further interior shots with quick release.
- **Late Clock Interior Setup Shot** – Interior shots with extended possession in late shot-clock scenarios.
- **Interior Setup Shot** – General interior shots with longer touch time.
- **Late Clock Closely Guarded Jumper** – Perimeter shots under heavy defensive pressure and low shot clock.
- **Closely Guarded Jumper** – General perimeter jumpers with tight defense.
- **Catch-and-Shoot** – Perimeter shots taken immediately upon receiving a pass.
- **Setup Jumper** – Perimeter shots taken after some setup movement.

Each zone has its own **logistic regression model**, ensuring that feature influences are tailored to different shot types, improving calibration within each category.

### 1️⃣ Compute Shot Similarity

- Each shot is compared to a player's **historical shots** using features:
  - `SHOT_DIST`, `CLOSE_DEF_DIST`, `DRIBBLES`, `TOUCH_TIME`
- Similarity is computed using a **Gaussian-based similarity function**.

This similarity determines how much a past shot influences the prediction for a new shot. If a new shot closely resembles previous attempts, those past shots have **greater weight** in influencing the probability calculation. Conversely, **low similarity means past shots contribute less to the new prediction**.

### 2️⃣ Optimize Prior Strength & Feature Weights (Optuna)

- Hyperparameters **prior_strength** (influence of base probability) and **feature importances** for similarities are optimized using **Optuna**.
- This balances historical performance influence with league-wide priors.

### 3️⃣ Apply Similarity-Based Adjustment

- **If a player takes a rare shot (low similarity to past shots), its probability is penalized**.
- This adjustment prevents **overrating players on shots they rarely take**. For example, **DeAndre Jordan's three-point attempts** would otherwise be overrated because most of his past shots are highly successful dunks and layups, giving those past shots too much influence on unrelated attempts.

### 4️⃣ Regularization via Box-Cox Transformations

- A **Box-Cox transformation** ensures similarity scores scale appropriately and prevents extreme overcorrections.

### 5️⃣ Final Model: Logistic Regression

- A **logistic regression model** integrates the similarity adjustments and generates final shot probability predictions.

## Evaluation

We compare the **player-agnostic model**, **player-adjusted model**, and **final similarity-weighted model** using **log loss**.
