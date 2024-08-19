%%  清空环境变量
warning off             % 关闭报警信息
clear                   % 清空变量
close all
%%  读取数据

data='E:\分类\NEW\4_7_NEW\3res50avg10_col_tex_2class.mat' ;   %读取数据的路径

data=cell2mat(struct2cell(load(data)));  %读取数据

%%  分析数据                          
% res = res(randperm(num_res), :);          % 打乱数据集（不打乱数据时，注释该行）
flag_conusion = 2;                        % 标志位为1，打开混淆矩阵
num_class = length(unique(data(:, end)));
%%  设置变量存储数据
P_train = []; P_test = [];
T_train = []; T_test = [];

%%  划分数据集
%     mid_res = res((res(:, end) == i), :);           % 循环取出不同类别的样本
%     mid_size = size(mid_res, 1);                    % 得到不同类别样本个数
%     mid_tiran = round(num_size * mid_size);         % 得到该类别的训练样本个数
for j=1:size(data,2)-1
    data_biao{1,j}=['特征',num2str(j)];
end
data_biao{1,size(data,2)}='预测值';
A_data=data(1:20,:);
A_data1=data;
data_biao1=data_biao;

num_runs = 20;  % 运行次数
total_accuracy = 0;  % 用于累加所有运行的Accuracy
total_precision = 0;  % 用于累加所有运行的Precision
total_recall = 0;  % 用于累加所有运行的Recall
total_f1_score = 0;  % 用于累加所有运行的F1-score
class_accuracies_sum = zeros((length(unique(data(:, end)))), 1);
a=50;
%10/70(3_3)
b = 500;
for i = 1:num_runs
    select_feature_num=b;   %特征选择的个数
    print_index_name=[];
    RF_Model = TreeBagger(300,A_data(:,1:end-1),A_data(:,end),'Method','regression','OOBPredictorImportance','on');
    imp = RF_Model.OOBPermutedPredictorDeltaError;
    [~,sort_feature]=sort(imp,'descend');
    index_name=data_biao1;
    feature_need_last=sort_feature(1:select_feature_num);
    for NN=1:length(feature_need_last)
        print_index_name{1,NN}=index_name{1,feature_need_last(NN)};
    end

    data_select=[A_data1(:,feature_need_last),A_data1(:,end)];  %经过特征选择后的数据
    res=data_select;
    %%  划分训练集和测试集
    P_train = [res(1: 20, 1: end - 1)]';       % 训练集输入
    T_train = [res(1: 20, end)]';              % 训练集输出

    P_test  = [res(21: end, 1: end - 1)]';  % 测试集输入
    T_test  = [res(21: end, end)]';         % 测试集输出
    M = size(P_train, 2);
    N = size(P_test , 2);
    %%  数据归一化
    [p_train, ps_input] = mapminmax(P_train, 0, 1);
    p_test = mapminmax('apply', P_test, ps_input );
    t_train = T_train;
    t_test  = T_test ;

    %%  转置以适应模型
    p_train = p_train'; p_test = p_test';
    t_train = t_train'; t_test = t_test';

    %%  参数设置
    pso_option.c1      = 1;                    % c1:初始为1.5, pso参数局部搜索能力  1.5
    pso_option.c2      = 10;                    % c2:初始为1.7, pso参数全局搜索能力  10
    pso_option.maxgen  = 100;                    % maxgen:最大进化数量设置为100  100
    pso_option.sizepop =  5;                     % sizepop:种群最大数量设置为5  5
    pso_option.k  = 0.6;                         % 初始为0.6(k belongs to [0.1,1.0]),速率和x的关系(V = kX)
    pso_option.wV = 1;                           % wV:初始为1(wV best belongs to [0.8,1.2]),速率更新公式中速度前面的弹性系数
    pso_option.wP = 1;                           % wP:初始为1,种群更新公式中速度前面的弹性系数
    pso_option.v  = 3;                           % v:初始为3,SVM Cross Validation参数

    pso_option.popcmax = 100;                    % popcmax:初始为100, SVM 参数c的变化的最大值.  100
    pso_option.popcmin = 0.1;                    % popcmin:初始为0.1, SVM 参数c的变化的最小值.  0.1
    pso_option.popgmax = 1;                    % popgmax:初始为100, SVM 参数g的变化的最大值.  0.1
    pso_option.popgmin = 0.001;                    % popgmin:初始为0.1, SVM 参数g的变化的最小值.  0.001

    %%  提取最佳参数c和g
    [bestacc, bestc, bestg] = pso_svm_class(t_train, p_train, pso_option);

    %%  建立模型  1/0.0028  1/0.0458
    cmd = [' -c ', num2str(bestc), ' -g ', num2str(bestg)];
    model = svmtrain(t_train, p_train, cmd);

    %%  仿真测试
    T_sim1 = svmpredict(t_train, p_train, model);
    T_sim2 = svmpredict(t_test , p_test , model);

    %%  数据排序
    [T_train, index_1] = sort(T_train);
    [T_test , index_2] = sort(T_test );

    T_sim1 = T_sim1(index_1);
    T_sim2 = T_sim2(index_2);

    %%  性能评价
    error1 = sum((T_sim1' == T_train)) / M * 100 ;
    error2 = sum((T_sim2' == T_test )) / N * 100 ;
    % 计算混淆矩阵
    C = confusionmat(T_test, T_sim2');

    % 计算Accuracy
    accuracy = sum(diag(C)) / sum(C, 'all');

    % 计算Precision
    precision = diag(C) ./ sum(C, 1)';

    % 计算Recall
    recall = diag(C) ./ sum(C, 2);

    % 计算F1-score
    f1_score = 2 * precision .* recall ./ (precision + recall);

    % 累加指标
    total_accuracy = total_accuracy + accuracy;
    total_precision = total_precision + mean(precision);
    total_recall = total_recall + mean(recall);
    total_f1_score = total_f1_score + mean(f1_score);

    % 计算每一类的分类准确率并累加到数组中
    for j = 1:size(C, 1)
        class_accuracies_sum(j) = class_accuracies_sum(j) + C(j, j) / sum(C(j, :));
    end
end
%%

% 计算每一类的平均分类准确率
average_class_accuracies = class_accuracies_sum / num_runs;

% 输出每一类的平均分类准确率
disp(['Average Class Accuracies: ', mat2str(average_class_accuracies)]);
% 计算平均指标
mean_accuracy = total_accuracy / num_runs;
mean_precision = total_precision / num_runs;
mean_recall = total_recall / num_runs;
mean_f1_score = total_f1_score / num_runs;

disp(['Mean Accuracy: ', num2str(mean_accuracy)]);
disp(['Mean Precision: ', num2str(mean_precision)]);
disp(['Mean Recall: ', num2str(mean_recall)]);
disp(['Mean F1-score: ', num2str(mean_f1_score)]);
