
clear
warning off
close all
%%  ��ȡ����

data='E:\����\NEW\4_7_NEW\3col_tex_res50avg7_3class.mat' ;   %��ȡ���ݵ�·��

data=cell2mat(struct2cell(load(data)));  %��ȡ����

%%  ��������                          
% res = res(randperm(num_res), :);          % �������ݼ�������������ʱ��ע�͸��У�
flag_conusion = 1;                        % ��־λΪ1���򿪻�������Ҫ��2018�汾�����ϣ�
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
%400/400 500/500
for i = 1:num_runs
    select_feature_num=50;   %����ѡ��ĸ���
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
    num_dim = size(res, 2) - 1;

    P_train = [res(1: 20, 1: end - 1)];       % ѵ��������
    T_train = [res(1: 20, end)];              % ѵ�������

    P_test  = [res(21: end, 1: end - 1)];  % ���Լ�����
    T_test  = [res(21: end, end)];         % ���Լ����

    %%  ����ת��
    P_train = P_train'; P_test = P_test';
    T_train = T_train'; T_test = T_test';

    %%  �õ�ѵ�����Ͳ�����������
    M = size(P_train, 2);
    N = size(P_test , 2);

    %%  ���ݹ�һ��
    [P_train, ps_input] = mapminmax(P_train, 0, 1);
    P_test  = mapminmax('apply', P_test, ps_input);

    t_train =  categorical(T_train)';
    t_test  =  categorical(T_test )';

    %%  ����ƽ��
    %   ������ƽ�̳�1ά����ֻ��һ�ִ���ʽ
    %   Ҳ����ƽ�̳�2ά���ݣ��Լ�3ά���ݣ���Ҫ�޸Ķ�Ӧģ�ͽṹ
    %   ����Ӧ��ʼ�պ���������ݽṹ����һ��
    p_train =  double(reshape(P_train, num_dim, 1, 1, M));
    p_test  =  double(reshape(P_test , num_dim, 1, 1, N));

    %%  ��������ṹ
    layers = [
        imageInputLayer([num_dim, 1, 1])                           % �����

        convolution2dLayer([2, 1], 16, 'Padding', 'same')          % ����˴�СΪ 2*1 ����16�����
        batchNormalizationLayer                                    % ����һ����
        reluLayer                                                  % relu �����

        maxPooling2dLayer([2, 1], 'Stride', [2, 1])                % ���ػ��� ��СΪ 2*1 ����Ϊ [2, 1]

        convolution2dLayer([2, 1], 32, 'Padding', 'same')          % ����˴�СΪ 2*1 ����32�����
        batchNormalizationLayer                                    % ����һ����
        reluLayer                                                  % relu �����

        fullyConnectedLayer(num_class)                             % ȫ���Ӳ㣨�������
        softmaxLayer                                               % ��ʧ������
        classificationLayer];                                      % �����
    a=1000;
    b=a-50;
    %%  ��������
    options = trainingOptions('adam', ...      % Adam �ݶ��½��㷨
        'MaxEpochs', a, ...                  % ���ѵ������ 500
        'InitialLearnRate', 1e-3, ...          % ��ʼѧϰ��Ϊ 0.001
        'L2Regularization', 1e-2, ...          % L2���򻯲���
        'LearnRateSchedule', 'piecewise', ...  % ѧϰ���½�
        'LearnRateDropFactor', 0.1, ...        % ѧϰ���½����� 0.1
        'LearnRateDropPeriod', b, ...        % ����450��ѵ���� ѧϰ��Ϊ 0.001 * 0.1
        'Shuffle', 'every-epoch', ...          % ÿ��ѵ���������ݼ�
        'ValidationPatience', 100, ...         % �ر���֤
        'Plots', 'training-progress', ...      % ��������
        'Verbose', false);

    %%  ѵ��ģ��
    net = trainNetwork(p_train, t_train, layers, options);

    %%  Ԥ��ģ��
    t_sim1 = predict(net, p_train);
    t_sim2 = predict(net, p_test );

    %%  ����һ��
    T_sim1 = vec2ind(t_sim1');
    T_sim2 = vec2ind(t_sim2');

    %%  ��������
    error1 = sum((T_sim1 == T_train)) / M * 100 ;
    error2 = sum((T_sim2 == T_test )) / N * 100;
    % �����������
    C = confusionmat(T_test, T_sim2');

    % ����Accuracy
    accuracy = sum(diag(C)) / sum(C, 'all')

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
