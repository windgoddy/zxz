clear;

data_str='E:\分类\NEW\4_7_NEW\3res50avg10_col_tex_2class.mat' ;    %500 600

data=cell2mat(struct2cell(load(data_str)));  %读取数据

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

b=700; %特征选择的个数
for i = 1:num_runs
    select_feature_num=b;   
    print_index_name=[];
    RF_Model = TreeBagger(50,A_data(:,1:end-1),A_data(:,end),'Method','regression','OOBPredictorImportance','on'); %特征选择随机森林棵树 
    imp = RF_Model.OOBPermutedPredictorDeltaError;
    [~,sort_feature]=sort(imp,'descend');
    index_name=data_biao1;
    feature_need_last=sort_feature(1:select_feature_num);
    for NN=1:length(feature_need_last)
        print_index_name{1,NN}=index_name{1,feature_need_last(NN)};
    end

    data_select=[A_data1(:,feature_need_last),A_data1(:,end)];  %经过特征选择后的数据

    x_feature_label=data_select(:,1:end-1);    %x特征
    y_feature_label=data_select(:,end);          %y标签
    index_label=1:94;  % 数据索引

    %训练集，验证集，测试集
    train_x_feature_label=x_feature_label(index_label(1:a),:);
    train_y_feature_label=y_feature_label(index_label(1:a),:);
    test_x_feature_label=x_feature_label(index_label((a+1):end),:);
    test_y_feature_label=y_feature_label(index_label((a+1):end),:);
    %Zscore 标准化
    %训练集
    x_mu = mean(train_x_feature_label);  x_sig = std(train_x_feature_label);
    train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;    % 训练数据标准化
    y_mu = mean(train_y_feature_label);  y_sig = std(train_y_feature_label);
    train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化
    %测试集
    test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;    % 测试数据标准化
    test_y_feature_label_norm = (test_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化

    num_tree = 300;   % 集成树的棵树
    Mdl = TreeBagger(num_tree, train_x_feature_label_norm, train_y_feature_label, 'Method', 'classification');

    % 训练集预测结果
    y_train_predict = RF_process(predict(Mdl, train_x_feature_label_norm));

    % 测试集预测结果
    y_test_predict = RF_process(predict(Mdl, test_x_feature_label_norm));

    % 其他代码...

    % 计算混淆矩阵
    C = confusionmat(test_y_feature_label, y_test_predict);

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