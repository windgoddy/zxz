% Deep feature extraction

load('net_avg000001_256.mat')
imdsTest = imageDatastore('E:\分类\Testing set','IncludeSubfolders',true,'LabelSource','foldernames');
imdsTrain = imageDatastore('E:\分类\Training set','IncludeSubfolders',true,'LabelSource','foldernames');

net = trainedNetwork_1;
inputSize = net.Layers(1).InputSize;
n_feature1 = 'resnet50_avgpool000001_300_Train';
n_feature2 = 'resnet50_avgpool000001_300_Test';

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
layer = 'avg_pool';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');
n_feature = double(vertcat(featuresTrain,featuresTest));

save([n_feature1,'.mat'],'featuresTrain');
save([n_feature2,'.mat'],'featuresTest');


%% 

X1 = load('E:\分类\NEW\4_7_NEW\burn_train_3col.mat'); 
X2 = load('E:\分类\NEW\4_7_NEW\burn_test_3col.mat');

X3 = load('E:\分类\NEW\4_7_NEW\resnet50_avgpool000001_300_Train.mat'); 
X4 = load('E:\分类\NEW\4_7_NEW\resnet50_avgpool000001_300_Test.mat');

X6 = load('E:\分类\NEW\4_7_NEW\burn_train_tex.mat'); 
X7 = load('E:\分类\NEW\4_7_NEW\burn_test_tex.mat');
% X6 = load('E:\分类\NEW\3_10_NEW\burn_train_tex.mat'); 
% X7 = load('E:\分类\NEW\burn_test_tex.mat');

label = load('E:\分类\NEW\label.mat');
label1 = load('E:\分类\NEW\label1.mat');
label3 = load('E:\分类\NEW\label3.mat');
label31 = load('E:\分类\NEW\label31.mat');

X_class_c1=[X1.COL,X6.FTS,X3.featuresTrain,label.label];
X_class_c2=[X2.COL,X7.FTS,X4.featuresTest,label1.label1];

X_class_c3=[X3.featuresTrain,X1.COL,X6.FTS,label3.label];
X_class_c4=[X4.featuresTest,X2.COL,X7.FTS,label31.label1];

X_class_ct = double(vertcat(X_class_c1, X_class_c2));
X_class_ct2 = double(vertcat(X_class_c3, X_class_c4));

save('333col_tex_res50avg000001_256_3class.mat','X_class_ct');
save('333res50avg000001_256_col_tex_2class.mat','X_class_ct2');

fprintf('done!\n');

