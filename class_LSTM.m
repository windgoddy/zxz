%% 初始化
clear
warning off
close all
%%  读取数据

data='E:\分类\NEW\4_7_NEW\3col_tex_res50avg7_3class.mat' ;   %读取数据的路径

data=cell2mat(struct2cell(load(data)));  %读取数据

%%  分析数据                          
% res = res(randperm(num_res), :);          % 打乱数据集（不打乱数据时，注释该行）
flag_conusion = 1;                        % 标志位为1，打开混淆矩阵（要求2018版本及以上）
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
a=20;
%400/400 500/500
b=100;
for i = 1:num_runs
    select_feature_num=b;   %特征选择的个数
    print_index_name=[];
    RF_Model = TreeBagger(400,A_data(:,1:end-1),A_data(:,end),'Method','regression','OOBPredictorImportance','on');
    imp = RF_Model.OOBPermutedPredictorDeltaError;
    [~,sort_feature]=sort(imp,'descend');
    index_name=data_biao1;
    feature_need_last=sort_feature(1:select_feature_num);
    for NN=1:length(feature_need_last)
        print_index_name{1,NN}=index_name{1,feature_need_last(NN)};
    end

    data_select=[A_data1(:,feature_need_last),A_data1(:,end)];  %经过特征选择后的数据
    res=data_select;
    num_dim = size(res, 2) - 1;

    P_train = [res(1: 20, 1: end - 1)];       % 训练集输入
    T_train = [res(1: 20, end)];              % 训练集输出

    P_test  = [res(21: end, 1: end - 1)];  % 测试集输入
    T_test  = [res(21: end, end)];         % 测试集输出

    %%  数据转置
    P_train = P_train'; P_test = P_test';
    T_train = T_train'; T_test = T_test';

    %%  得到训练集和测试样本个数
    M = size(P_train, 2);
    N = size(P_test , 2);

    %%  数据归一化
    [P_train, ps_input] = mapminmax(P_train, 0, 1);
    P_test  = mapminmax('apply', P_test, ps_input);

    t_train =  categorical(T_train)';
    t_test  =  categorical(T_test )';

    %%  数据平铺
    %   将数据平铺成1维数据只是一种处理方式
    %   也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
    %   但是应该始终和输入层数据结构保持一致
    P_train =  double(reshape(P_train, num_dim, 1, 1, M));
    P_test  =  double(reshape(P_test , num_dim, 1, 1, N));

    % t_train = t_train';
    % t_test  = t_test' ;
    % p_train =  double(reshape(P_train, num_dim, 1, 1, M));
    % p_test  =  double(reshape(P_test , num_dim, 1, 1, N));

    %%  数据格式转换
    for i = 1 : M
        p_train{i, 1} = P_train(:, :, 1, i);
    end

    for i = 1 : N
        p_test{i, 1}  = P_test( :, :, 1, i);
    end
    %%  建立模型
    layers = [
        sequenceInputLayer(num_dim)                                  % 输入层

        lstmLayer(6, 'OutputMode', 'last')                      % LSTM层
        reluLayer                                               % Relu激活层

        fullyConnectedLayer(num_class)                             % 全连接层（类别数）
        softmaxLayer                                            % 分类层
        classificationLayer];

    %%  参数设置
    options = trainingOptions('adam', ...       % Adam 梯度下降算法
        'MaxEpochs', 1000, ...                  % 最大迭代次数
        'InitialLearnRate', 0.01, ...           % 初始学习率
        'LearnRateSchedule', 'piecewise', ...   % 学习率下降
        'LearnRateDropFactor', 0.1, ...         % 学习率下降因子
        'LearnRateDropPeriod', 900, ...         % 经过 750 次训练后 学习率为 0.01 * 0.1
        'Shuffle', 'every-epoch', ...           % 每次训练打乱数据集
        'ValidationPatience', Inf, ...          % 关闭验证
        'L2Regularization', 1e-4, ...           % 正则化参数
        'Plots', 'training-progress', ...       % 画出曲线
        'Verbose', false);
    %%  训练模型
    net = trainNetwork(p_train, t_train, layers, options);

    %%  预测模型
    t_sim1 = predict(net, p_train);
    t_sim2 = predict(net, p_test );

    %%  反归一化
    T_sim1 = vec2ind(t_sim1');
    T_sim2 = vec2ind(t_sim2');

    %%  性能评价
    error1 = sum((T_sim1 == T_train)) / M * 100 ;
    error2 = sum((T_sim2 == T_test )) / N * 100 ;
    % 计算混淆矩阵
    C = confusionmat(T_test, T_sim2');

    % 计算Accuracy
    accuracy = sum(diag(C)) / sum(C, 'all')

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
