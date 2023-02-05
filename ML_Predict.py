import argparse

## 指定预测年份,算法
description = 'Description ' \
              'year is the year you want to predict\n' \
              'algorithm is the method you want to apply, including "ridge","rf" and "xgb" \n' \
              'train is used to update the params in xgb, otherwise the pretuned params will be used'


def get_parser():
    parser = argparse.ArgumentParser(description= description)
    parser.add_argument('-y','--year',default=2022,type=int)
    parser.add_argument('-a','--algorithm',default='xgb',type=str)
    parser.add_argument('-t','--train',default='False',type=str)
    return parser

from ML_Pre_Treat import get_train_test_data, show_pre_table, show_mse_error





from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


from sklearn.model_selection import GridSearchCV




class Algorithm():

    def __init__(self):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    ## 岭回归
    def reg_test(self):
        reg = Ridge(alpha=0.1)
        reg.fit(train_x, train_y)

        pred_result = reg.predict(test_x)

        return pred_result

    ## 随机森林
    def rf_test(self):
        rf = RandomForestRegressor(n_estimators=300)
        rf.fit(train_x,train_y)
        pred_result = rf.predict(test_x)

        return pred_result

    ## xgboost
    def xgb_test(self,cv_train=False):
        DM_train = xgb.DMatrix(train_x, train_y)
        DM_test = xgb.DMatrix(test_x, test_y)

        # params = {
        #     'colsample_bytree': 0.7,
        #     'gamma': 0,
        #     'learning_rate': 0.05,
        #     'max_depth': 5,
        #     'min_child_weight': 4,
        #     'subsample': 0.7
        # }
        #
        # xg_reg = xgb.train(params=params, dtrain=DM_train, num_boost_round=100)
        # pred_result = xg_reg.predict(DM_test)
        #
        # return pred_result

        if cv_train == True:
            model = xgb.XGBRegressor()
            grid = {
                'max_depth': [4, 5, 6],
                'gamma': [0, 1, 2, 3],
                'learning_rate': [0.05, 0.1, 0.2],
                'min_child_weight': [1, 2, 3, 4],
                'subsample': [0.7, 1],  # 行的有放回采样，70%
                'colsample_bytree': [0.7, 1],  # 列无放回采样， 90%
            }

            reg_grid = GridSearchCV(estimator=model, param_grid=grid, scoring='neg_mean_squared_error', cv=5, verbose=1)

            reg_grid.fit(train_x, train_y)

            params = reg_grid.best_params_

        else:
            params = {
                'colsample_bytree': 0.7,
                'gamma': 0,
                'learning_rate': 0.05,
                'max_depth': 5,
                'min_child_weight': 4,
                'subsample': 0.7
            }

        xg_reg = xgb.train(params=params, dtrain=DM_train, num_boost_round=100)
        pred_result = xg_reg.predict(DM_test)

        return pred_result


if __name__ == '__main__':
    ## 获取 train_x, train_y
    parser = get_parser()
    args = parser.parse_args()
    pred_year = args.year
    al = args.algorithm
    train = args.train

    train_x, train_y, test_x, test_y,test = get_train_test_data(pred_year)
    method = Algorithm()
    if al == "ridge":
        result = method.reg_test()
        show_pre_table(result,test_y=test_y,test=test)
    elif al == "rf":
        result = method.rf_test()
        show_pre_table(result,test_y=test_y,test=test)
    elif al == "xgb":
        result = method.xgb_test(cv_train=train)
        show_pre_table(result,test_y=test_y,test=test)
    else:
        print("This algorithm is not provided.")


