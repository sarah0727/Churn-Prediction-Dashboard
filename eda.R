library(shiny)
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(e1071)
library(nnet)
library(ggplot2)

# Load and preprocess the dataset
data <- read.csv("C:/Users/User/OneDrive/Documents/Predictive Analytics/dashboard/Churn_Modelling.csv")
train_data <- data[sample(nrow(data), 0.8*nrow(data)), ]
test_data <- data[setdiff(1:nrow(data), 1:nrow(train_data)), ]

# Train the models
model_lr <- train(Exited ~ ., data=train_data, method="glm", family="binomial")
model_rf <- randomForest(Exited ~ ., data=train_data, ntree=500)
model_xgb <- xgboost(data=as.matrix(train_data[, -which(names(train_data) == "Exited")]), 
                     label=train_data$Exited, max_depth=3, eta=0.1, nrounds=100)
model_svm <- svm(Exited ~ ., data=train_data, kernel="radial", cost=10, gamma=0.1)
model_nn <- nnet(Exited ~ ., data=train_data, size=5, decay=0.01, maxit=100)

ui <- fluidPage(
  titlePanel("Churn Prediction Model Comparison"),
  sidebarLayout(
    sidebarPanel(
      selectInput("model", "Select Model", c("Logistic Regression", "Random Forest", "XGBoost", "SVM", "Neural Network")),
      selectInput("metric", "Select Metric", c("Accuracy", "Precision", "Recall", "F1-Score"))
    ),
    mainPanel(
      plotOutput("feature_importance"),
      plotOutput("model_comparison"),
      tableOutput("model_metrics")
    )
  )
)

server <- function(input, output) {
  # Feature importance plot
  output$feature_importance <- renderPlot({
    model <- get(paste0("model_", tolower(gsub(" ", "_", input$model))))
    if (class(model) == "train") {
      varImp(model) %>% 
        ggplot(aes(x=reorder(Variables, Importance), y=Importance)) +
        geom_bar(stat="identity") +
        coord_flip() +
        labs(x="Feature", y="Importance")
    } else if (class(model) == "randomForest") {
      importance(model) %>% 
        data.frame() %>%
        rownames_to_column("Feature") %>%
        ggplot(aes(x=reorder(Feature, `%IncMSE`), y=`%IncMSE`)) +
        geom_bar(stat="identity") +
        coord_flip() +
        labs(x="Feature", y="Importance")
    }
  })
  
  # Model comparison plot
  output$model_comparison <- renderPlot({
    model_performance <- data.frame(
      Model=c("Logistic Regression", "Random Forest", "XGBoost", "SVM", "Neural Network"),
      Accuracy=c(accuracy(predict(model_lr, test_data), test_data$Exited),
                 accuracy(predict(model_rf, test_data), test_data$Exited),
                 accuracy(predict(model_xgb, test_data), test_data$Exited),
                 accuracy(predict(model_svm, test_data), test_data$Exited),
                 accuracy(predict(model_nn, test_data), test_data$Exited))
    )
    
    ggplot(model_performance, aes(x=reorder(Model, Accuracy), y=Accuracy)) +
      geom_bar(stat="identity") +
      coord_flip() +
      labs(x="Model", y=input$metric)
  })
  
  # Model metrics table
  output$model_metrics <- renderTable({
    model <- get(paste0("model_", tolower(gsub(" ", "_", input$model))))
    if (class(model) == "train") {
      data.frame(
        Accuracy=round(accuracy(predict(model, test_data), test_data$Exited), 3),
        Precision=round(precision(predict(model, test_data), test_data$Exited), 3),
        Recall=round(recall(predict(model, test_data), test_data$Exited), 3),
        `F1-Score`=round(f_meas(predict(model, test_data), test_data$Exited), 3)
      )
    } else {
      data.frame(
        Accuracy=round(mean(predict(model, test_data) == test_data$Exited), 3),
        Precision=round(precision(predict(model, test_data), test_data$Exited), 3),
        Recall=round(recall(predict(model, test_data), test_data$Exited), 3),
        `F1-Score`=round(f_meas(predict(model, test_data), test_data$Exited), 3)
      )
    }
  })
}

shinyApp(ui, server)

