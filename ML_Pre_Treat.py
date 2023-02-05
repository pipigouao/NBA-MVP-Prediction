import argparse

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error


def get_train_test_data(pred_year):
    ## 填补缺失值
    stats = pd.read_csv("player_mvp_stats.csv", index_col=0)

    stats["GB"] = stats["GB"].str.replace("—", "0")
    stats["GB"] = pd.to_numeric(stats["GB"])

    stats = stats.apply(pd.to_numeric, errors='ignore')
    stats = stats.fillna(0)

    ## 选择预测变量
    predictors = ["Age", "G", "GS", "MP", "FG", "FGA", 'FG%', '3P', '3PA'
        , '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%'
        , 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'W', 'L', 'W/L%',
                  'GB', 'PS/G', 'PA/G', 'SRS', 'Pos']

    ## 创建哑变量
    pd.get_dummies(stats['Pos'],prefix='Pos')
    dummy_pos = pd.get_dummies(stats['Pos'],prefix='Pos')
    predictors.remove("Pos")
    predictors_and_dum = predictors + dummy_pos.columns.tolist()

    stats_and_dum = stats.join(dummy_pos)

    ## 2.加入ratio变量 （可删除）
    stats_and_dum[["PTS_R", "AST_R", "STL_R", "BLK_R", "3P_R"]] = stats_and_dum.groupby("Year")[["PTS", "AST", "STL", "BLK", "3P"]].apply(lambda x: x/x.mean())
    new_predictor = ["PTS_R", "AST_R", "STL_R", "BLK_R", "3P_R"]
    predictors_and_dum += new_predictor

    ## 3.划分 train/test
    train = stats_and_dum[~(stats_and_dum["Year"] == pred_year)]
    test = stats_and_dum[stats_and_dum["Year"] == pred_year]

    train_x = train[predictors_and_dum]
    train_y = train["Share"]

    test_x = test[predictors_and_dum]
    test_y = test["Share"]

    return train_x, train_y, test_x, test_y,test


## 展示预测结果
def show_pre_table(mvp_prediction,test_y,test):
    pd_mvp_pred = pd.DataFrame(mvp_prediction, columns=["predictions"], index=test_y.index)
    combinations = pd.concat([test[["Player", "Share"]], pd_mvp_pred], axis=1)

    actual = combinations.sort_values("Share", ascending=False)
    predicted = combinations.sort_values("predictions", ascending=False)
    actual["Rk"] = list(range(1, actual.shape[0] + 1))
    predicted["Predicted_Rk"] = list(range(1, predicted.shape[0] + 1))

    pred_table = actual.merge(predicted, on="Player")

    # 删除多余的列并修改列名
    del pred_table["Share_y"]
    del pred_table["predictions_y"]
    pred_table.rename(columns={'Share_x': 'Share', 'predictions_x': 'predictions'}, inplace=True)

    pred_table["Rk_diff"] = pred_table["Rk"] - pred_table["Predicted_Rk"]

    print(pred_table)

    # return pred_table


'''
展示预测误差
1.不指定topn时，计算全部误差
2.指定topn时，计算前实际前十的误差
'''


def show_mse_error(pred_table, topn=0):
    if topn == 0:
        mse = mean_squared_error(pred_table["Share"], pred_table["predictions"])
    else:
        topN = pred_table.iloc[:10, :]
        mse = mean_squared_error(topN["Share"], topN["predictions"])

    return mse





