模型分为三个部分

1.1 数据爬取

可更参数有两个：
1.爬取年份 list
2.chrome 模拟器路径（根据自己电脑），需安装对应Selenium模块
目前存储了 1990~2022年数据

可以获取对应年份的 mvps, players, teams 数据，存储在对应名称的csv文件中

1.2 数据更新
可更参数：爬取年份list
nba reference 网站容易封号，因此最好不要一次性爬太多数据
设置好爬取年份list后，同 1.1 数据爬取，会获取对应年份的 mvps, players, teams 数据
而后直接加在原本的csv文件后面，完成数据更新（注意不要加入重复数据）
注：一旦加入重复数据，可手动删除，或者回到 1.1模块，单独运行生成csv的部分，此部分不需要重复爬取网页，只需读取对应文件夹中的数据

2 数据清洗
无可更参数
清洗数据并合并 mvps, players, teams，获得 palyer_mvp_stats.csv 
此数据将用于最后一步机器学习的预测

3 机器学习预测
可更参数：预测年份（1年）
模型会把除这一年份外的数据用作训练集

自带函数：（具体参照代码中注释）
1.show_pre_table
2.show_mse_error
用于结果展示及比较

目前需改进部分：
1.随机森林参数调优
2.xgboost参数调优


目前模块三已较为冗长，如需加入数据探索及可视化部分可另建 2.5数据可视化
文中采用的各项指标解释说明可参考原网站


