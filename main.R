# Load required libraries
library(tidyverse)
library(caret)
library(ggplot2)
library(plotly)
library(shiny)
library(shinydashboard)
library(reshape2)
library(gridExtra)
library(pROC)

# Load data
data <- read.csv("C:/Users/User/OneDrive/Documents/Predictive Analytics/dashboard/Churn_Modelling.csv")

# Preprocess data
data <- data %>%
  select(-c(RowNumber, CustomerId, Surname)) %>%
  mutate(
    Gender = ifelse(Gender == "Male", 1, 0),
    Geography_France = ifelse(Geography == "France", 1, 0),
    Geography_Spain = ifelse(Geography == "Spain", 1, 0),
    Geography_Germany = ifelse(Geography == "Germany", 1, 0)
  ) %>%
  select(-Geography)

data <- data %>%
  mutate(
    NumOfProducts = abs(NumOfProducts),
    Balance = abs(Balance),
    HasCrCard = ifelse(HasCrCard < 0, 0, HasCrCard),
    IsActiveMember = ifelse(IsActiveMember < 0, 0, IsActiveMember),
    EstimatedSalary = ifelse(EstimatedSalary < 0, 0, EstimatedSalary),
    Exited = factor(Exited, levels = c(0, 1), labels = c("No", "Yes"))
  )
numeric_cols <- c("CreditScore", "Age", "Balance", "EstimatedSalary", "Tenure")
data[numeric_cols] <- scale(data[numeric_cols])

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data$Exited, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Train multiple models
models <- list()
conf_matrices <- list()
roc_curves <- list()
precision_data <- data.frame(Model = character(), Precision = numeric())
recall_data <- data.frame(Model = character(), Recall = numeric())

# Logistic Regression
models[["Logistic"]] <- train(Exited ~ ., data = trainData, method = "glm", family = "binomial")
predictions <- predict(models[["Logistic"]], testData)
conf <- confusionMatrix(predictions, testData$Exited)
conf_matrices[["Logistic"]] <- conf
precision_data <- rbind(precision_data, data.frame(Model = "Logistic", Precision = conf$byClass["Precision"]))
recall_data <- rbind(recall_data, data.frame(Model = "Logistic", Recall = conf$byClass["Recall"]))
roc_curves[["Logistic"]] <- roc(as.numeric(testData$Exited), as.numeric(predictions))

# Decision Tree
models[["Decision Tree"]] <- train(Exited ~ ., data = trainData, method = "rpart")
predictions <- predict(models[["Decision Tree"]], testData)
conf <- confusionMatrix(predictions, testData$Exited)
conf_matrices[["Decision Tree"]] <- conf
precision_data <- rbind(precision_data, data.frame(Model = "Decision Tree", Precision = conf$byClass["Precision"]))
recall_data <- rbind(recall_data, data.frame(Model = "Decision Tree", Recall = conf$byClass["Recall"]))
roc_curves[["Decision Tree"]] <- roc(as.numeric(testData$Exited), as.numeric(predictions))

# Random Forest
models[["Random Forest"]] <- train(Exited ~ ., data = trainData, method = "rf")
predictions <- predict(models[["Random Forest"]], testData)
conf <- confusionMatrix(predictions, testData$Exited)
conf_matrices[["Random Forest"]] <- conf
precision_data <- rbind(precision_data, data.frame(Model = "Random Forest", Precision = conf$byClass["Precision"]))
recall_data <- rbind(recall_data, data.frame(Model = "Random Forest", Recall = conf$byClass["Recall"]))
roc_curves[["Random Forest"]] <- roc(as.numeric(testData$Exited), as.numeric(predictions))

# Gradient Boosting
models[["Gradient Boosting"]] <- train(Exited ~ ., data = trainData, method = "gbm", verbose = FALSE)
predictions <- predict(models[["Gradient Boosting"]], testData)
conf <- confusionMatrix(predictions, testData$Exited)
conf_matrices[["Gradient Boosting"]] <- conf
precision_data <- rbind(precision_data, data.frame(Model = "Gradient Boosting", Precision = conf$byClass["Precision"]))
recall_data <- rbind(recall_data, data.frame(Model = "Gradient Boosting", Recall = conf$byClass["Recall"]))
roc_curves[["Gradient Boosting"]] <- roc(as.numeric(testData$Exited), as.numeric(predictions))

# Support Vector Machine
models[["SVM"]] <- train(Exited ~ ., data = trainData, method = "svmRadial")
predictions <- predict(models[["SVM"]], testData)
conf <- confusionMatrix(predictions, testData$Exited)
conf_matrices[["SVM"]] <- conf
precision_data <- rbind(precision_data, data.frame(Model = "SVM", Precision = conf$byClass["Precision"]))
recall_data <- rbind(recall_data, data.frame(Model = "SVM", Recall = conf$byClass["Recall"]))
roc_curves[["SVM"]] <- roc(as.numeric(testData$Exited), as.numeric(predictions))

# Collect model accuracies
accuracies <- data.frame(
  Model = names(models),
  Accuracy = sapply(conf_matrices, function(cm) cm$overall["Accuracy"])
)

# Define UI for the dashboard
ui <- dashboardPage(
  dashboardHeader(title = "Churn Prediction Model Dashboard"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Model Overview", tabName = "overview", icon = icon("table")),
      menuItem("Model Comparison", tabName = "comparison", icon = icon("chart-line")),
      menuItem("Metrics Evaluation", tabName = "metrics", icon = icon("th"))
    )
  ),
  
  dashboardBody(
    tabItems(
      # Tab 1: Model Overview
      tabItem(tabName = "overview",
              fluidRow(
                box(title = "Missing Values Summary", status = "primary", solidHeader = TRUE, collapsible = TRUE, width = 6,
                    tableOutput("missingTable")),
                box(title = "Target Variable Distribution", status = "primary", solidHeader = TRUE, collapsible = TRUE, width = 6,
                    plotOutput("targetPieChart")),
                box(title = "Numeric Features Distribution", status = "primary", solidHeader = TRUE, collapsible = TRUE, width = 6,
                    plotOutput("numericDistPlot")),
                box(title = "Correlation Matrix", status = "primary", solidHeader = TRUE, collapsible = TRUE, width = 6,
                    plotOutput("correlationMatrixPlot"))
              )),
      
      # Tab 2: Model Comparison
      tabItem(tabName = "comparison",
              fluidRow(
                box(title = "Model Accuracy Comparison", status = "primary", solidHeader = TRUE, collapsible = TRUE, width = 12,
                    plotOutput("accuracyLinePlot")),
                box(title = "Model Precision Comparison", status = "primary", solidHeader = TRUE, collapsible = TRUE, width = 6,
                    plotOutput("precisionBarPlot")),
                box(title = "Model Recall Comparison", status = "primary", solidHeader = TRUE, collapsible = TRUE, width = 6,
                    plotOutput("recallBarPlot"))
              )),
      
      # Tab 3: Metrics Evaluation
      tabItem(tabName = "metrics",
              fluidRow(
                box(title = "Confusion Matrices for Each Model", status = "info", solidHeader = TRUE, collapsible = TRUE, width = 6,
                    plotOutput("confusionMatrixPlots")),
                box(title = "ROC Curves for Each Model", status = "info", solidHeader = TRUE, collapsible = TRUE, width = 6,
                    plotOutput("rocCurvePlots"))
              ))
    )
  )
)

server <- function(input, output) {
  
  # Missing Values Table
  output$missingTable <- renderTable({
    data.frame(Feature = names(data), MissingValues = colSums(is.na(data)))
  })
  
  # Target Variable Pie Chart
  output$targetPieChart <- renderPlot({
    ggplot(data, aes(x = "", fill = Exited)) +
      geom_bar(width = 1) +
      coord_polar(theta = "y") +
      labs(title = "Target Variable (Exited) Distribution") +
      theme_minimal()
  })
  
  # Numeric Features Distribution
  output$numericDistPlot <- renderPlot({
    data %>%
      gather(key = "Feature", value = "Value", numeric_cols) %>%
      ggplot(aes(x = Value, fill = Feature)) +
      geom_histogram(bins = 30, alpha = 0.7) +
      facet_wrap(~ Feature, scales = "free") +
      labs(title = "Distribution of Numeric Features") +
      theme_minimal()
  })
  
  # Correlation Matrix Plot
  output$correlationMatrixPlot <- renderPlot({
    corr <- cor(data[numeric_cols])
    melted_corr <- melt(corr)
    ggplot(melted_corr, aes(x = Var1, y = Var2, fill = value)) +
      geom_tile() +
      scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
      labs(title = "Correlation Matrix") +
      theme_minimal()
  })
  
  # Accuracy Line Plot
  output$accuracyLinePlot <- renderPlot({
    ggplot(accuracies, aes(x = Model, y = Accuracy, group = 1)) +
      geom_line() +
      geom_point() +
      labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
      theme_minimal()
  })
  
  # Precision Bar Plot
  output$precisionBarPlot <- renderPlot({
    ggplot(precision_data, aes(x = Model, y = Precision, fill = Model)) +
      geom_bar(stat = "identity") +
      labs(title = "Model Precision Comparison", x = "Model", y = "Precision") +
      theme_minimal()
  })
  
  # Recall Bar Plot
  output$recallBarPlot <- renderPlot({
    ggplot(recall_data, aes(x = Model, y = Recall, fill = Model)) +
      geom_bar(stat = "identity") +
      labs(title = "Model Recall Comparison", x = "Model", y = "Recall") +
      theme_minimal()
  })
  
  # Confusion Matrix Plots
  output$confusionMatrixPlots <- renderPlot({
    plots <- lapply(names(conf_matrices), function(model) {
      cm <- as.data.frame(conf_matrices[[model]]$table)
      cm$Model <- model
      ggplot(cm, aes(x = Reference, y = Prediction, fill = Freq)) +
        geom_tile() +
        geom_text(aes(label = Freq)) +
        scale_fill_gradient(low = "white", high = "blue") +
        labs(title = paste("Confusion Matrix:", model)) +
        theme_minimal()
    })
    do.call(grid.arrange, c(plots, nrow = 3))
  })
  
  # ROC Curve Plots
  output$rocCurvePlots <- renderPlot({
    plots <- lapply(names(roc_curves), function(model) {
      roc_data <- roc_curves[[model]]
      ggroc(roc_data) +
        labs(title = paste("ROC Curve:", model)) +
        theme_minimal()
    })
    do.call(grid.arrange, c(plots, nrow = 3))
  })
}

# Run the Shiny App
shinyApp(ui = ui, server = server)

