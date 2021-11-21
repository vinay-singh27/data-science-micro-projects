#setting up optuna
from sklearn import metrics
from lightgbm import LGBMRegressor
import optuna
from optuna import samplers

def hyperparameter_tuning(X_train, y_train, X_val, y_val) :

    samplers.TPESampler()

    def objective(trial):

        param = {
            "metric": "rmse",
            "n_estimators" : trial.suggest_int("n_estimators", 200, 1200),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
            'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02]),
            'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
            'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
            'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
        }
        
        model = LGBMRegressor(**param)  

        model.fit(X_train, y_train, eval_set=[(X_val,y_val)], early_stopping_rounds=100, verbose=False)
        preds = model.predict(X_val)
        rmse = metrics.mean_squared_error(y_val, preds,squared=False)
        
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    print('Best score:', study.best_trial.params)

    return study.best_params