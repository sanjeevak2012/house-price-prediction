
# Ridge Param Grid
ridge_param_grid = {
    'alpha': [0.1, 1.0, 10.0],  # Controls how much the model is "penalized" to prevent overfitting (smaller alpha = less penalty)
    'fit_intercept': [True, False], # Determines if the model should try to find a starting point on the y-axis (like in y = mx + c, where 'c' is the intercept)
}

elasticnet_param_grid = {
    'alpha': [0.1, 1.0, 10.0],  # How much to shrink the effect of less important features (higher alpha means more shrinking)
    'l1_ratio': [0.2, 0.5, 0.8],  # Controls the blend between two ways of shrinking: 0 means mostly one way (Ridge), 1 means mostly the other (Lasso)
    'fit_intercept': [True, False],  # Should the model include a starting point (intercept) like in y = mx + c? (True to include, False to not)
    # 'max_iter': [1000, 5000, 10000] # How many times the model should try to improve its fit before stopping
}

xgb_param_grid = {
    'n_estimators': [100, 200, 300],  # How many individual decision trees to build
    'learning_rate': [0.01, 0.1, 0.2],  # How much each tree's prediction impacts the overall result (smaller means more trees needed)
    'max_depth': [3, 4, 5],  # The maximum number of levels in each decision tree
    'min_child_weight': [1, 2, 3],  # The minimum number of data points required in a final branch of a tree
    'subsample': [0.8, 0.9, 1.0],  # What percentage of the data to use for building each tree (helps prevent overfitting)
    'colsample_bytree': [0.8, 0.9, 1.0],  # What percentage of the features (columns) to use for building each tree
    'gamma': [0, 0.1, 0.2],  # A threshold for creating new branches in a tree (higher means less complex trees)
    'reg_alpha': [0, 0.1, 1.0],  # A penalty applied to large feature weights (L1 regularization - encourages simpler models)
    'reg_lambda': [0, 0.1, 1.0],  # Another penalty applied to large feature weights (L2 regularization - keeps weights small)
}