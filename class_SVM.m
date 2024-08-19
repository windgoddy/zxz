%%  ��ջ�������
warning off             % �رձ�����Ϣ
clear                   % ��ձ���
close all
%%  ��ȡ����

data='E:\����\NEW\4_7_NEW\3res50avg10_col_tex_2class.mat' ;   %��ȡ���ݵ�·��

data=cell2mat(struct2cell(load(data)));  %��ȡ����

%%  ��������                          
% res = res(randperm(num_res), :);          % �������ݼ�������������ʱ��ע�͸��У�
flag_conusion = 2;                        % ��־λΪ1���򿪻�������
num_class = length(unique(data(:, end)));
%%  ���ñ����洢����
P_train = []; P_test = [];
T_train = []; T_test = [];

%%  �������ݼ�
%     mid_res = res((res(:, end) == i), :);           % ѭ��ȡ����ͬ��������
%     mid_size = size(mid_res, 1);                    % �õ���ͬ�����������
%     mid_tiran = round(num_size * mid_size);         % �õ�������ѵ����������
for j=1:size(data,2)-1
    data_biao{1,j}=['����',num2str(j)];
end
data_biao{1,size(data,2)}='Ԥ��ֵ';
A_data=data(1:20,:);
A_data1=data;
data_biao1=data_biao;

num_runs = 20;  % ���д���
total_accuracy = 0;  % �����ۼ��������е�Accuracy
total_precision = 0;  % �����ۼ��������е�Precision
total_recall = 0;  % �����ۼ��������е�Recall
total_f1_score = 0;  % �����ۼ��������е�F1-score
class_accuracies_sum = zeros((length(unique(data(:, end)))), 1);
a=50;
%10/70(3_3)
b = 500;
for i = 1:num_runs
    select_feature_num=b;   %����ѡ��ĸ���
    print_index_name=[];
    RF_Model = TreeBagger(300,A_data(:,1:end-1),A_data(:,end),'Method','regression','OOBPredictorImportance','on');
    imp = RF_Model.OOBPermutedPredictorDeltaError;
    [~,sort_feature]=sort(imp,'descend');
    index_name=data_biao1;
    feature_need_last=sort_feature(1:select_feature_num);
    for NN=1:length(feature_need_last)
        print_index_name{1,NN}=index_name{1,feature_need_last(NN)};
    end

    data_select=[A_data1(:,feature_need_last),A_data1(:,end)];  %��������ѡ��������
    res=data_select;
    %%  ����ѵ�����Ͳ��Լ�
    P_train = [res(1: 20, 1: end - 1)]';       % ѵ��������
    T_train = [res(1: 20, end)]';              % ѵ�������

    P_test  = [res(21: end, 1: end - 1)]';  % ���Լ�����
    T_test  = [res(21: end, end)]';         % ���Լ����
    M = size(P_train, 2);
    N = size(P_test , 2);
    %%  ���ݹ�һ��
    [p_train, ps_input] = mapminmax(P_train, 0, 1);
    p_test = mapminmax('apply', P_test, ps_input );
    t_train = T_train;
    t_test  = T_test ;

    %%  ת������Ӧģ��
    p_train = p_train'; p_test = p_test';
    t_train = t_train'; t_test = t_test';

    %%  ��������
    pso_option.c1      = 1;                    % c1:��ʼΪ1.5, pso�����ֲ���������  1.5
    pso_option.c2      = 10;                    % c2:��ʼΪ1.7, pso����ȫ����������  10
    pso_option.maxgen  = 100;                    % maxgen:��������������Ϊ100  100
    pso_option.sizepop =  5;                     % sizepop:��Ⱥ�����������Ϊ5  5
    pso_option.k  = 0.6;                         % ��ʼΪ0.6(k belongs to [0.1,1.0]),���ʺ�x�Ĺ�ϵ(V = kX)
    pso_option.wV = 1;                           % wV:��ʼΪ1(wV best belongs to [0.8,1.2]),���ʸ��¹�ʽ���ٶ�ǰ��ĵ���ϵ��
    pso_option.wP = 1;                           % wP:��ʼΪ1,��Ⱥ���¹�ʽ���ٶ�ǰ��ĵ���ϵ��
    pso_option.v  = 3;                           % v:��ʼΪ3,SVM Cross Validation����

    pso_option.popcmax = 100;                    % popcmax:��ʼΪ100, SVM ����c�ı仯�����ֵ.  100
    pso_option.popcmin = 0.1;                    % popcmin:��ʼΪ0.1, SVM ����c�ı仯����Сֵ.  0.1
    pso_option.popgmax = 1;                    % popgmax:��ʼΪ100, SVM ����g�ı仯�����ֵ.  0.1
    pso_option.popgmin = 0.001;                    % popgmin:��ʼΪ0.1, SVM ����g�ı仯����Сֵ.  0.001

    %%  ��ȡ��Ѳ���c��g
    [bestacc, bestc, bestg] = pso_svm_class(t_train, p_train, pso_option);

    %%  ����ģ��  1/0.0028  1/0.0458
    cmd = [' -c ', num2str(bestc), ' -g ', num2str(bestg)];
    model = svmtrain(t_train, p_train, cmd);

    %%  �������
    T_sim1 = svmpredict(t_train, p_train, model);
    T_sim2 = svmpredict(t_test , p_test , model);

    %%  ��������
    [T_train, index_1] = sort(T_train);
    [T_test , index_2] = sort(T_test );

    T_sim1 = T_sim1(index_1);
    T_sim2 = T_sim2(index_2);

    %%  ��������
    error1 = sum((T_sim1' == T_train)) / M * 100 ;
    error2 = sum((T_sim2' == T_test )) / N * 100 ;
    % �����������
    C = confusionmat(T_test, T_sim2');

    % ����Accuracy
    accuracy = sum(diag(C)) / sum(C, 'all');

    % ����Precision
    precision = diag(C) ./ sum(C, 1)';

    % ����Recall
    recall = diag(C) ./ sum(C, 2);

    % ����F1-score
    f1_score = 2 * precision .* recall ./ (precision + recall);

    % �ۼ�ָ��
    total_accuracy = total_accuracy + accuracy;
    total_precision = total_precision + mean(precision);
    total_recall = total_recall + mean(recall);
    total_f1_score = total_f1_score + mean(f1_score);

    % ����ÿһ��ķ���׼ȷ�ʲ��ۼӵ�������
    for j = 1:size(C, 1)
        class_accuracies_sum(j) = class_accuracies_sum(j) + C(j, j) / sum(C(j, :));
    end
end
%%

% ����ÿһ���ƽ������׼ȷ��
average_class_accuracies = class_accuracies_sum / num_runs;

% ���ÿһ���ƽ������׼ȷ��
disp(['Average Class Accuracies: ', mat2str(average_class_accuracies)]);
% ����ƽ��ָ��
mean_accuracy = total_accuracy / num_runs;
mean_precision = total_precision / num_runs;
mean_recall = total_recall / num_runs;
mean_f1_score = total_f1_score / num_runs;

disp(['Mean Accuracy: ', num2str(mean_accuracy)]);
disp(['Mean Precision: ', num2str(mean_precision)]);
disp(['Mean Recall: ', num2str(mean_recall)]);
disp(['Mean F1-score: ', num2str(mean_f1_score)]);
