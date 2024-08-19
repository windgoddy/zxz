%% 创建和训练深度学习模型
% 该脚本可创建和训练具有以下属性的深度学习网络:
%%
% 
%  层数: 180
%  连接数: 195
%  训练设置文件: E:\分类\NEW\github\res50_train.mat
%
%% 
% 运行此脚本以创建网络层、导入训练和验证数据并训练网络。网络层存储在工作区变量 |lgraph| 中。经过训练的网络存储在工作区变量 |net| 中。
% 
% 要了解详细信息，请参阅 <matlab:helpview('deeplearning','generate_matlab_code') 从深度网络设计器生成 
% MATLAB 代码>。
% 
% 由 MATLAB 于 2024-08-19 10:29:21 自动生成
%% 加载初始参数
% 加载用于网络初始化的参数。对于迁移学习，网络初始化参数是初始预训练网络的参数。

trainingSetup = load("E:\分类\NEW\github\res50_train.mat");
%% 导入数据
% 导入训练和验证数据。

imdsTrain = imageDatastore("E:\分类\Training set1","IncludeSubfolders",true,"LabelSource","foldernames");
imdsValidation = imageDatastore("E:\分类\Testing set1","IncludeSubfolders",true,"LabelSource","foldernames");
%% 增强设置

imageAugmenter = imageDataAugmenter(...
    "RandRotation",[0 180],...
    "RandScale",[1 10],...
    "RandXReflection",true);

% 调整图像大小以匹配网络输入层。
augimdsTrain = augmentedImageDatastore([224 224 3],imdsTrain,"DataAugmentation",imageAugmenter);
augimdsValidation = augmentedImageDatastore([224 224 3],imdsValidation);
%% 设置训练选项
% 指定训练时要使用的选项。

opts = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.01,...
    "MaxEpochs",50,...
    "Shuffle","every-epoch",...
    "Plots","training-progress",...
    "ValidationData",augimdsValidation);
%% 创建层次图
% 创建层次图变量以包含网络层。

lgraph = layerGraph();
%% 添加层分支
% 将网络分支添加到层次图中。每个分支均为一个线性层组。

tempLayers = [
    imageInputLayer([224 224 3],"Name","input_1","Mean",trainingSetup.input_1.Mean)
    convolution2dLayer([7 7],64,"Name","conv1","Padding",[3 3 3 3],"Stride",[2 2],"Bias",trainingSetup.conv1.Bias,"Weights",trainingSetup.conv1.Weights)
    batchNormalizationLayer("Name","bn_conv1","Epsilon",0.001,"Offset",trainingSetup.bn_conv1.Offset,"Scale",trainingSetup.bn_conv1.Scale,"TrainedMean",trainingSetup.bn_conv1.TrainedMean,"TrainedVariance",trainingSetup.bn_conv1.TrainedVariance)
    reluLayer("Name","activation_1_relu")
    maxPooling2dLayer([3 3],"Name","max_pooling2d_1","Padding",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2a_branch2a","BiasLearnRateFactor",0,"Bias",trainingSetup.res2a_branch2a.Bias,"Weights",trainingSetup.res2a_branch2a.Weights)
    batchNormalizationLayer("Name","bn2a_branch2a","Epsilon",0.001,"Offset",trainingSetup.bn2a_branch2a.Offset,"Scale",trainingSetup.bn2a_branch2a.Scale,"TrainedMean",trainingSetup.bn2a_branch2a.TrainedMean,"TrainedVariance",trainingSetup.bn2a_branch2a.TrainedVariance)
    reluLayer("Name","activation_2_relu")
    convolution2dLayer([3 3],64,"Name","res2a_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",trainingSetup.res2a_branch2b.Bias,"Weights",trainingSetup.res2a_branch2b.Weights)
    batchNormalizationLayer("Name","bn2a_branch2b","Epsilon",0.001,"Offset",trainingSetup.bn2a_branch2b.Offset,"Scale",trainingSetup.bn2a_branch2b.Scale,"TrainedMean",trainingSetup.bn2a_branch2b.TrainedMean,"TrainedVariance",trainingSetup.bn2a_branch2b.TrainedVariance)
    reluLayer("Name","activation_3_relu")
    convolution2dLayer([1 1],256,"Name","res2a_branch2c","BiasLearnRateFactor",0,"Bias",trainingSetup.res2a_branch2c.Bias,"Weights",trainingSetup.res2a_branch2c.Weights)
    batchNormalizationLayer("Name","bn2a_branch2c","Epsilon",0.001,"Offset",trainingSetup.bn2a_branch2c.Offset,"Scale",trainingSetup.bn2a_branch2c.Scale,"TrainedMean",trainingSetup.bn2a_branch2c.TrainedMean,"TrainedVariance",trainingSetup.bn2a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res2a_branch1","BiasLearnRateFactor",0,"Bias",trainingSetup.res2a_branch1.Bias,"Weights",trainingSetup.res2a_branch1.Weights)
    batchNormalizationLayer("Name","bn2a_branch1","Epsilon",0.001,"Offset",trainingSetup.bn2a_branch1.Offset,"Scale",trainingSetup.bn2a_branch1.Scale,"TrainedMean",trainingSetup.bn2a_branch1.TrainedMean,"TrainedVariance",trainingSetup.bn2a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_1")
    reluLayer("Name","activation_4_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2b_branch2a","BiasLearnRateFactor",0,"Bias",trainingSetup.res2b_branch2a.Bias,"Weights",trainingSetup.res2b_branch2a.Weights)
    batchNormalizationLayer("Name","bn2b_branch2a","Epsilon",0.001,"Offset",trainingSetup.bn2b_branch2a.Offset,"Scale",trainingSetup.bn2b_branch2a.Scale,"TrainedMean",trainingSetup.bn2b_branch2a.TrainedMean,"TrainedVariance",trainingSetup.bn2b_branch2a.TrainedVariance)
    reluLayer("Name","activation_5_relu")
    convolution2dLayer([3 3],64,"Name","res2b_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",trainingSetup.res2b_branch2b.Bias,"Weights",trainingSetup.res2b_branch2b.Weights)
    batchNormalizationLayer("Name","bn2b_branch2b","Epsilon",0.001,"Offset",trainingSetup.bn2b_branch2b.Offset,"Scale",trainingSetup.bn2b_branch2b.Scale,"TrainedMean",trainingSetup.bn2b_branch2b.TrainedMean,"TrainedVariance",trainingSetup.bn2b_branch2b.TrainedVariance)
    reluLayer("Name","activation_6_relu")
    convolution2dLayer([1 1],256,"Name","res2b_branch2c","BiasLearnRateFactor",0,"Bias",trainingSetup.res2b_branch2c.Bias,"Weights",trainingSetup.res2b_branch2c.Weights)
    batchNormalizationLayer("Name","bn2b_branch2c","Epsilon",0.001,"Offset",trainingSetup.bn2b_branch2c.Offset,"Scale",trainingSetup.bn2b_branch2c.Scale,"TrainedMean",trainingSetup.bn2b_branch2c.TrainedMean,"TrainedVariance",trainingSetup.bn2b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_2")
    reluLayer("Name","activation_7_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2c_branch2a","BiasLearnRateFactor",0,"Bias",trainingSetup.res2c_branch2a.Bias,"Weights",trainingSetup.res2c_branch2a.Weights)
    batchNormalizationLayer("Name","bn2c_branch2a","Epsilon",0.001,"Offset",trainingSetup.bn2c_branch2a.Offset,"Scale",trainingSetup.bn2c_branch2a.Scale,"TrainedMean",trainingSetup.bn2c_branch2a.TrainedMean,"TrainedVariance",trainingSetup.bn2c_branch2a.TrainedVariance)
    reluLayer("Name","activation_8_relu")
    convolution2dLayer([3 3],64,"Name","res2c_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",trainingSetup.res2c_branch2b.Bias,"Weights",trainingSetup.res2c_branch2b.Weights)
    batchNormalizationLayer("Name","bn2c_branch2b","Epsilon",0.001,"Offset",trainingSetup.bn2c_branch2b.Offset,"Scale",trainingSetup.bn2c_branch2b.Scale,"TrainedMean",trainingSetup.bn2c_branch2b.TrainedMean,"TrainedVariance",trainingSetup.bn2c_branch2b.TrainedVariance)
    reluLayer("Name","activation_9_relu")
    convolution2dLayer([1 1],256,"Name","res2c_branch2c","BiasLearnRateFactor",0,"Bias",trainingSetup.res2c_branch2c.Bias,"Weights",trainingSetup.res2c_branch2c.Weights)
    batchNormalizationLayer("Name","bn2c_branch2c","Epsilon",0.001,"Offset",trainingSetup.bn2c_branch2c.Offset,"Scale",trainingSetup.bn2c_branch2c.Scale,"TrainedMean",trainingSetup.bn2c_branch2c.TrainedMean,"TrainedVariance",trainingSetup.bn2c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_3")
    reluLayer("Name","activation_10_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3a_branch2a","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",trainingSetup.res3a_branch2a.Bias,"Weights",trainingSetup.res3a_branch2a.Weights)
    batchNormalizationLayer("Name","bn3a_branch2a","Epsilon",0.001,"Offset",trainingSetup.bn3a_branch2a.Offset,"Scale",trainingSetup.bn3a_branch2a.Scale,"TrainedMean",trainingSetup.bn3a_branch2a.TrainedMean,"TrainedVariance",trainingSetup.bn3a_branch2a.TrainedVariance)
    reluLayer("Name","activation_11_relu")
    convolution2dLayer([3 3],128,"Name","res3a_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",trainingSetup.res3a_branch2b.Bias,"Weights",trainingSetup.res3a_branch2b.Weights)
    batchNormalizationLayer("Name","bn3a_branch2b","Epsilon",0.001,"Offset",trainingSetup.bn3a_branch2b.Offset,"Scale",trainingSetup.bn3a_branch2b.Scale,"TrainedMean",trainingSetup.bn3a_branch2b.TrainedMean,"TrainedVariance",trainingSetup.bn3a_branch2b.TrainedVariance)
    reluLayer("Name","activation_12_relu")
    convolution2dLayer([1 1],512,"Name","res3a_branch2c","BiasLearnRateFactor",0,"Bias",trainingSetup.res3a_branch2c.Bias,"Weights",trainingSetup.res3a_branch2c.Weights)
    batchNormalizationLayer("Name","bn3a_branch2c","Epsilon",0.001,"Offset",trainingSetup.bn3a_branch2c.Offset,"Scale",trainingSetup.bn3a_branch2c.Scale,"TrainedMean",trainingSetup.bn3a_branch2c.TrainedMean,"TrainedVariance",trainingSetup.bn3a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res3a_branch1","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",trainingSetup.res3a_branch1.Bias,"Weights",trainingSetup.res3a_branch1.Weights)
    batchNormalizationLayer("Name","bn3a_branch1","Epsilon",0.001,"Offset",trainingSetup.bn3a_branch1.Offset,"Scale",trainingSetup.bn3a_branch1.Scale,"TrainedMean",trainingSetup.bn3a_branch1.TrainedMean,"TrainedVariance",trainingSetup.bn3a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_4")
    reluLayer("Name","activation_13_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3b_branch2a","BiasLearnRateFactor",0,"Bias",trainingSetup.res3b_branch2a.Bias,"Weights",trainingSetup.res3b_branch2a.Weights)
    batchNormalizationLayer("Name","bn3b_branch2a","Epsilon",0.001,"Offset",trainingSetup.bn3b_branch2a.Offset,"Scale",trainingSetup.bn3b_branch2a.Scale,"TrainedMean",trainingSetup.bn3b_branch2a.TrainedMean,"TrainedVariance",trainingSetup.bn3b_branch2a.TrainedVariance)
    reluLayer("Name","activation_14_relu")
    convolution2dLayer([3 3],128,"Name","res3b_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",trainingSetup.res3b_branch2b.Bias,"Weights",trainingSetup.res3b_branch2b.Weights)
    batchNormalizationLayer("Name","bn3b_branch2b","Epsilon",0.001,"Offset",trainingSetup.bn3b_branch2b.Offset,"Scale",trainingSetup.bn3b_branch2b.Scale,"TrainedMean",trainingSetup.bn3b_branch2b.TrainedMean,"TrainedVariance",trainingSetup.bn3b_branch2b.TrainedVariance)
    reluLayer("Name","activation_15_relu")
    convolution2dLayer([1 1],512,"Name","res3b_branch2c","BiasLearnRateFactor",0,"Bias",trainingSetup.res3b_branch2c.Bias,"Weights",trainingSetup.res3b_branch2c.Weights)
    batchNormalizationLayer("Name","bn3b_branch2c","Epsilon",0.001,"Offset",trainingSetup.bn3b_branch2c.Offset,"Scale",trainingSetup.bn3b_branch2c.Scale,"TrainedMean",trainingSetup.bn3b_branch2c.TrainedMean,"TrainedVariance",trainingSetup.bn3b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_5")
    reluLayer("Name","activation_16_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3c_branch2a","BiasLearnRateFactor",0,"Bias",trainingSetup.res3c_branch2a.Bias,"Weights",trainingSetup.res3c_branch2a.Weights)
    batchNormalizationLayer("Name","bn3c_branch2a","Epsilon",0.001,"Offset",trainingSetup.bn3c_branch2a.Offset,"Scale",trainingSetup.bn3c_branch2a.Scale,"TrainedMean",trainingSetup.bn3c_branch2a.TrainedMean,"TrainedVariance",trainingSetup.bn3c_branch2a.TrainedVariance)
    reluLayer("Name","activation_17_relu")
    convolution2dLayer([3 3],128,"Name","res3c_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",trainingSetup.res3c_branch2b.Bias,"Weights",trainingSetup.res3c_branch2b.Weights)
    batchNormalizationLayer("Name","bn3c_branch2b","Epsilon",0.001,"Offset",trainingSetup.bn3c_branch2b.Offset,"Scale",trainingSetup.bn3c_branch2b.Scale,"TrainedMean",trainingSetup.bn3c_branch2b.TrainedMean,"TrainedVariance",trainingSetup.bn3c_branch2b.TrainedVariance)
    reluLayer("Name","activation_18_relu")
    convolution2dLayer([1 1],512,"Name","res3c_branch2c","BiasLearnRateFactor",0,"Bias",trainingSetup.res3c_branch2c.Bias,"Weights",trainingSetup.res3c_branch2c.Weights)
    batchNormalizationLayer("Name","bn3c_branch2c","Epsilon",0.001,"Offset",trainingSetup.bn3c_branch2c.Offset,"Scale",trainingSetup.bn3c_branch2c.Scale,"TrainedMean",trainingSetup.bn3c_branch2c.TrainedMean,"TrainedVariance",trainingSetup.bn3c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_6")
    reluLayer("Name","activation_19_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3d_branch2a","BiasLearnRateFactor",0,"Bias",trainingSetup.res3d_branch2a.Bias,"Weights",trainingSetup.res3d_branch2a.Weights)
    batchNormalizationLayer("Name","bn3d_branch2a","Epsilon",0.001,"Offset",trainingSetup.bn3d_branch2a.Offset,"Scale",trainingSetup.bn3d_branch2a.Scale,"TrainedMean",trainingSetup.bn3d_branch2a.TrainedMean,"TrainedVariance",trainingSetup.bn3d_branch2a.TrainedVariance)
    reluLayer("Name","activation_20_relu")
    convolution2dLayer([3 3],128,"Name","res3d_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",trainingSetup.res3d_branch2b.Bias,"Weights",trainingSetup.res3d_branch2b.Weights)
    batchNormalizationLayer("Name","bn3d_branch2b","Epsilon",0.001,"Offset",trainingSetup.bn3d_branch2b.Offset,"Scale",trainingSetup.bn3d_branch2b.Scale,"TrainedMean",trainingSetup.bn3d_branch2b.TrainedMean,"TrainedVariance",trainingSetup.bn3d_branch2b.TrainedVariance)
    reluLayer("Name","activation_21_relu")
    convolution2dLayer([1 1],512,"Name","res3d_branch2c","BiasLearnRateFactor",0,"Bias",trainingSetup.res3d_branch2c.Bias,"Weights",trainingSetup.res3d_branch2c.Weights)
    batchNormalizationLayer("Name","bn3d_branch2c","Epsilon",0.001,"Offset",trainingSetup.bn3d_branch2c.Offset,"Scale",trainingSetup.bn3d_branch2c.Scale,"TrainedMean",trainingSetup.bn3d_branch2c.TrainedMean,"TrainedVariance",trainingSetup.bn3d_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_7")
    reluLayer("Name","activation_22_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4a_branch2a","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",trainingSetup.res4a_branch2a.Bias,"Weights",trainingSetup.res4a_branch2a.Weights)
    batchNormalizationLayer("Name","bn4a_branch2a","Epsilon",0.001,"Offset",trainingSetup.bn4a_branch2a.Offset,"Scale",trainingSetup.bn4a_branch2a.Scale,"TrainedMean",trainingSetup.bn4a_branch2a.TrainedMean,"TrainedVariance",trainingSetup.bn4a_branch2a.TrainedVariance)
    reluLayer("Name","activation_23_relu")
    convolution2dLayer([3 3],256,"Name","res4a_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",trainingSetup.res4a_branch2b.Bias,"Weights",trainingSetup.res4a_branch2b.Weights)
    batchNormalizationLayer("Name","bn4a_branch2b","Epsilon",0.001,"Offset",trainingSetup.bn4a_branch2b.Offset,"Scale",trainingSetup.bn4a_branch2b.Scale,"TrainedMean",trainingSetup.bn4a_branch2b.TrainedMean,"TrainedVariance",trainingSetup.bn4a_branch2b.TrainedVariance)
    reluLayer("Name","activation_24_relu")
    convolution2dLayer([1 1],1024,"Name","res4a_branch2c","BiasLearnRateFactor",0,"Bias",trainingSetup.res4a_branch2c.Bias,"Weights",trainingSetup.res4a_branch2c.Weights)
    batchNormalizationLayer("Name","bn4a_branch2c","Epsilon",0.001,"Offset",trainingSetup.bn4a_branch2c.Offset,"Scale",trainingSetup.bn4a_branch2c.Scale,"TrainedMean",trainingSetup.bn4a_branch2c.TrainedMean,"TrainedVariance",trainingSetup.bn4a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],1024,"Name","res4a_branch1","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",trainingSetup.res4a_branch1.Bias,"Weights",trainingSetup.res4a_branch1.Weights)
    batchNormalizationLayer("Name","bn4a_branch1","Epsilon",0.001,"Offset",trainingSetup.bn4a_branch1.Offset,"Scale",trainingSetup.bn4a_branch1.Scale,"TrainedMean",trainingSetup.bn4a_branch1.TrainedMean,"TrainedVariance",trainingSetup.bn4a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_8")
    reluLayer("Name","activation_25_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4b_branch2a","BiasLearnRateFactor",0,"Bias",trainingSetup.res4b_branch2a.Bias,"Weights",trainingSetup.res4b_branch2a.Weights)
    batchNormalizationLayer("Name","bn4b_branch2a","Epsilon",0.001,"Offset",trainingSetup.bn4b_branch2a.Offset,"Scale",trainingSetup.bn4b_branch2a.Scale,"TrainedMean",trainingSetup.bn4b_branch2a.TrainedMean,"TrainedVariance",trainingSetup.bn4b_branch2a.TrainedVariance)
    reluLayer("Name","activation_26_relu")
    convolution2dLayer([3 3],256,"Name","res4b_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",trainingSetup.res4b_branch2b.Bias,"Weights",trainingSetup.res4b_branch2b.Weights)
    batchNormalizationLayer("Name","bn4b_branch2b","Epsilon",0.001,"Offset",trainingSetup.bn4b_branch2b.Offset,"Scale",trainingSetup.bn4b_branch2b.Scale,"TrainedMean",trainingSetup.bn4b_branch2b.TrainedMean,"TrainedVariance",trainingSetup.bn4b_branch2b.TrainedVariance)
    reluLayer("Name","activation_27_relu")
    convolution2dLayer([1 1],1024,"Name","res4b_branch2c","BiasLearnRateFactor",0,"Bias",trainingSetup.res4b_branch2c.Bias,"Weights",trainingSetup.res4b_branch2c.Weights)
    batchNormalizationLayer("Name","bn4b_branch2c","Epsilon",0.001,"Offset",trainingSetup.bn4b_branch2c.Offset,"Scale",trainingSetup.bn4b_branch2c.Scale,"TrainedMean",trainingSetup.bn4b_branch2c.TrainedMean,"TrainedVariance",trainingSetup.bn4b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_9")
    reluLayer("Name","activation_28_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4c_branch2a","BiasLearnRateFactor",0,"Bias",trainingSetup.res4c_branch2a.Bias,"Weights",trainingSetup.res4c_branch2a.Weights)
    batchNormalizationLayer("Name","bn4c_branch2a","Epsilon",0.001,"Offset",trainingSetup.bn4c_branch2a.Offset,"Scale",trainingSetup.bn4c_branch2a.Scale,"TrainedMean",trainingSetup.bn4c_branch2a.TrainedMean,"TrainedVariance",trainingSetup.bn4c_branch2a.TrainedVariance)
    reluLayer("Name","activation_29_relu")
    convolution2dLayer([3 3],256,"Name","res4c_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",trainingSetup.res4c_branch2b.Bias,"Weights",trainingSetup.res4c_branch2b.Weights)
    batchNormalizationLayer("Name","bn4c_branch2b","Epsilon",0.001,"Offset",trainingSetup.bn4c_branch2b.Offset,"Scale",trainingSetup.bn4c_branch2b.Scale,"TrainedMean",trainingSetup.bn4c_branch2b.TrainedMean,"TrainedVariance",trainingSetup.bn4c_branch2b.TrainedVariance)
    reluLayer("Name","activation_30_relu")
    convolution2dLayer([1 1],1024,"Name","res4c_branch2c","BiasLearnRateFactor",0,"Bias",trainingSetup.res4c_branch2c.Bias,"Weights",trainingSetup.res4c_branch2c.Weights)
    batchNormalizationLayer("Name","bn4c_branch2c","Epsilon",0.001,"Offset",trainingSetup.bn4c_branch2c.Offset,"Scale",trainingSetup.bn4c_branch2c.Scale,"TrainedMean",trainingSetup.bn4c_branch2c.TrainedMean,"TrainedVariance",trainingSetup.bn4c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_10")
    reluLayer("Name","activation_31_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4d_branch2a","BiasLearnRateFactor",0,"Bias",trainingSetup.res4d_branch2a.Bias,"Weights",trainingSetup.res4d_branch2a.Weights)
    batchNormalizationLayer("Name","bn4d_branch2a","Epsilon",0.001,"Offset",trainingSetup.bn4d_branch2a.Offset,"Scale",trainingSetup.bn4d_branch2a.Scale,"TrainedMean",trainingSetup.bn4d_branch2a.TrainedMean,"TrainedVariance",trainingSetup.bn4d_branch2a.TrainedVariance)
    reluLayer("Name","activation_32_relu")
    convolution2dLayer([3 3],256,"Name","res4d_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",trainingSetup.res4d_branch2b.Bias,"Weights",trainingSetup.res4d_branch2b.Weights)
    batchNormalizationLayer("Name","bn4d_branch2b","Epsilon",0.001,"Offset",trainingSetup.bn4d_branch2b.Offset,"Scale",trainingSetup.bn4d_branch2b.Scale,"TrainedMean",trainingSetup.bn4d_branch2b.TrainedMean,"TrainedVariance",trainingSetup.bn4d_branch2b.TrainedVariance)
    reluLayer("Name","activation_33_relu")
    convolution2dLayer([1 1],1024,"Name","res4d_branch2c","BiasLearnRateFactor",0,"Bias",trainingSetup.res4d_branch2c.Bias,"Weights",trainingSetup.res4d_branch2c.Weights)
    batchNormalizationLayer("Name","bn4d_branch2c","Epsilon",0.001,"Offset",trainingSetup.bn4d_branch2c.Offset,"Scale",trainingSetup.bn4d_branch2c.Scale,"TrainedMean",trainingSetup.bn4d_branch2c.TrainedMean,"TrainedVariance",trainingSetup.bn4d_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_11")
    reluLayer("Name","activation_34_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4e_branch2a","BiasLearnRateFactor",0,"Bias",trainingSetup.res4e_branch2a.Bias,"Weights",trainingSetup.res4e_branch2a.Weights)
    batchNormalizationLayer("Name","bn4e_branch2a","Epsilon",0.001,"Offset",trainingSetup.bn4e_branch2a.Offset,"Scale",trainingSetup.bn4e_branch2a.Scale,"TrainedMean",trainingSetup.bn4e_branch2a.TrainedMean,"TrainedVariance",trainingSetup.bn4e_branch2a.TrainedVariance)
    reluLayer("Name","activation_35_relu")
    convolution2dLayer([3 3],256,"Name","res4e_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",trainingSetup.res4e_branch2b.Bias,"Weights",trainingSetup.res4e_branch2b.Weights)
    batchNormalizationLayer("Name","bn4e_branch2b","Epsilon",0.001,"Offset",trainingSetup.bn4e_branch2b.Offset,"Scale",trainingSetup.bn4e_branch2b.Scale,"TrainedMean",trainingSetup.bn4e_branch2b.TrainedMean,"TrainedVariance",trainingSetup.bn4e_branch2b.TrainedVariance)
    reluLayer("Name","activation_36_relu")
    convolution2dLayer([1 1],1024,"Name","res4e_branch2c","BiasLearnRateFactor",0,"Bias",trainingSetup.res4e_branch2c.Bias,"Weights",trainingSetup.res4e_branch2c.Weights)
    batchNormalizationLayer("Name","bn4e_branch2c","Epsilon",0.001,"Offset",trainingSetup.bn4e_branch2c.Offset,"Scale",trainingSetup.bn4e_branch2c.Scale,"TrainedMean",trainingSetup.bn4e_branch2c.TrainedMean,"TrainedVariance",trainingSetup.bn4e_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_12")
    reluLayer("Name","activation_37_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4f_branch2a","BiasLearnRateFactor",0,"Bias",trainingSetup.res4f_branch2a.Bias,"Weights",trainingSetup.res4f_branch2a.Weights)
    batchNormalizationLayer("Name","bn4f_branch2a","Epsilon",0.001,"Offset",trainingSetup.bn4f_branch2a.Offset,"Scale",trainingSetup.bn4f_branch2a.Scale,"TrainedMean",trainingSetup.bn4f_branch2a.TrainedMean,"TrainedVariance",trainingSetup.bn4f_branch2a.TrainedVariance)
    reluLayer("Name","activation_38_relu")
    convolution2dLayer([3 3],256,"Name","res4f_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",trainingSetup.res4f_branch2b.Bias,"Weights",trainingSetup.res4f_branch2b.Weights)
    batchNormalizationLayer("Name","bn4f_branch2b","Epsilon",0.001,"Offset",trainingSetup.bn4f_branch2b.Offset,"Scale",trainingSetup.bn4f_branch2b.Scale,"TrainedMean",trainingSetup.bn4f_branch2b.TrainedMean,"TrainedVariance",trainingSetup.bn4f_branch2b.TrainedVariance)
    reluLayer("Name","activation_39_relu")
    convolution2dLayer([1 1],1024,"Name","res4f_branch2c","BiasLearnRateFactor",0,"Bias",trainingSetup.res4f_branch2c.Bias,"Weights",trainingSetup.res4f_branch2c.Weights)
    batchNormalizationLayer("Name","bn4f_branch2c","Epsilon",0.001,"Offset",trainingSetup.bn4f_branch2c.Offset,"Scale",trainingSetup.bn4f_branch2c.Scale,"TrainedMean",trainingSetup.bn4f_branch2c.TrainedMean,"TrainedVariance",trainingSetup.bn4f_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_13")
    reluLayer("Name","activation_40_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5a_branch2a","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",trainingSetup.res5a_branch2a.Bias,"Weights",trainingSetup.res5a_branch2a.Weights)
    batchNormalizationLayer("Name","bn5a_branch2a","Epsilon",0.001,"Offset",trainingSetup.bn5a_branch2a.Offset,"Scale",trainingSetup.bn5a_branch2a.Scale,"TrainedMean",trainingSetup.bn5a_branch2a.TrainedMean,"TrainedVariance",trainingSetup.bn5a_branch2a.TrainedVariance)
    reluLayer("Name","activation_41_relu")
    convolution2dLayer([3 3],512,"Name","res5a_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",trainingSetup.res5a_branch2b.Bias,"Weights",trainingSetup.res5a_branch2b.Weights)
    batchNormalizationLayer("Name","bn5a_branch2b","Epsilon",0.001,"Offset",trainingSetup.bn5a_branch2b.Offset,"Scale",trainingSetup.bn5a_branch2b.Scale,"TrainedMean",trainingSetup.bn5a_branch2b.TrainedMean,"TrainedVariance",trainingSetup.bn5a_branch2b.TrainedVariance)
    reluLayer("Name","activation_42_relu")
    convolution2dLayer([1 1],2048,"Name","res5a_branch2c","BiasLearnRateFactor",0,"Bias",trainingSetup.res5a_branch2c.Bias,"Weights",trainingSetup.res5a_branch2c.Weights)
    batchNormalizationLayer("Name","bn5a_branch2c","Epsilon",0.001,"Offset",trainingSetup.bn5a_branch2c.Offset,"Scale",trainingSetup.bn5a_branch2c.Scale,"TrainedMean",trainingSetup.bn5a_branch2c.TrainedMean,"TrainedVariance",trainingSetup.bn5a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],2048,"Name","res5a_branch1","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",trainingSetup.res5a_branch1.Bias,"Weights",trainingSetup.res5a_branch1.Weights)
    batchNormalizationLayer("Name","bn5a_branch1","Epsilon",0.001,"Offset",trainingSetup.bn5a_branch1.Offset,"Scale",trainingSetup.bn5a_branch1.Scale,"TrainedMean",trainingSetup.bn5a_branch1.TrainedMean,"TrainedVariance",trainingSetup.bn5a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_14")
    reluLayer("Name","activation_43_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5b_branch2a","BiasLearnRateFactor",0,"Bias",trainingSetup.res5b_branch2a.Bias,"Weights",trainingSetup.res5b_branch2a.Weights)
    batchNormalizationLayer("Name","bn5b_branch2a","Epsilon",0.001,"Offset",trainingSetup.bn5b_branch2a.Offset,"Scale",trainingSetup.bn5b_branch2a.Scale,"TrainedMean",trainingSetup.bn5b_branch2a.TrainedMean,"TrainedVariance",trainingSetup.bn5b_branch2a.TrainedVariance)
    reluLayer("Name","activation_44_relu")
    convolution2dLayer([3 3],512,"Name","res5b_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",trainingSetup.res5b_branch2b.Bias,"Weights",trainingSetup.res5b_branch2b.Weights)
    batchNormalizationLayer("Name","bn5b_branch2b","Epsilon",0.001,"Offset",trainingSetup.bn5b_branch2b.Offset,"Scale",trainingSetup.bn5b_branch2b.Scale,"TrainedMean",trainingSetup.bn5b_branch2b.TrainedMean,"TrainedVariance",trainingSetup.bn5b_branch2b.TrainedVariance)
    reluLayer("Name","activation_45_relu")
    convolution2dLayer([1 1],2048,"Name","res5b_branch2c","BiasLearnRateFactor",0,"Bias",trainingSetup.res5b_branch2c.Bias,"Weights",trainingSetup.res5b_branch2c.Weights)
    batchNormalizationLayer("Name","bn5b_branch2c","Epsilon",0.001,"Offset",trainingSetup.bn5b_branch2c.Offset,"Scale",trainingSetup.bn5b_branch2c.Scale,"TrainedMean",trainingSetup.bn5b_branch2c.TrainedMean,"TrainedVariance",trainingSetup.bn5b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_15")
    reluLayer("Name","activation_46_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5c_branch2a","BiasLearnRateFactor",0,"Bias",trainingSetup.res5c_branch2a.Bias,"Weights",trainingSetup.res5c_branch2a.Weights)
    batchNormalizationLayer("Name","bn5c_branch2a","Epsilon",0.001,"Offset",trainingSetup.bn5c_branch2a.Offset,"Scale",trainingSetup.bn5c_branch2a.Scale,"TrainedMean",trainingSetup.bn5c_branch2a.TrainedMean,"TrainedVariance",trainingSetup.bn5c_branch2a.TrainedVariance)
    reluLayer("Name","activation_47_relu")
    convolution2dLayer([3 3],512,"Name","res5c_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",trainingSetup.res5c_branch2b.Bias,"Weights",trainingSetup.res5c_branch2b.Weights)
    batchNormalizationLayer("Name","bn5c_branch2b","Epsilon",0.001,"Offset",trainingSetup.bn5c_branch2b.Offset,"Scale",trainingSetup.bn5c_branch2b.Scale,"TrainedMean",trainingSetup.bn5c_branch2b.TrainedMean,"TrainedVariance",trainingSetup.bn5c_branch2b.TrainedVariance)
    reluLayer("Name","activation_48_relu")
    convolution2dLayer([1 1],2048,"Name","res5c_branch2c","BiasLearnRateFactor",0,"Bias",trainingSetup.res5c_branch2c.Bias,"Weights",trainingSetup.res5c_branch2c.Weights)
    batchNormalizationLayer("Name","bn5c_branch2c","Epsilon",0.001,"Offset",trainingSetup.bn5c_branch2c.Offset,"Scale",trainingSetup.bn5c_branch2c.Scale,"TrainedMean",trainingSetup.bn5c_branch2c.TrainedMean,"TrainedVariance",trainingSetup.bn5c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_16")
    reluLayer("Name","activation_49_relu")
    globalAveragePooling2dLayer("Name","avg_pool")
    convolution2dLayer([3 3],300,"Name","conv","Padding","same")
    batchNormalizationLayer("Name","batchnorm","Epsilon",0.0001)
    reluLayer("Name","relu")
    fullyConnectedLayer(3,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;
%% 连接层分支
% 连接网络的所有分支以创建网络图。

lgraph = connectLayers(lgraph,"max_pooling2d_1","res2a_branch2a");
lgraph = connectLayers(lgraph,"max_pooling2d_1","res2a_branch1");
lgraph = connectLayers(lgraph,"bn2a_branch1","add_1/in2");
lgraph = connectLayers(lgraph,"bn2a_branch2c","add_1/in1");
lgraph = connectLayers(lgraph,"activation_4_relu","res2b_branch2a");
lgraph = connectLayers(lgraph,"activation_4_relu","add_2/in2");
lgraph = connectLayers(lgraph,"bn2b_branch2c","add_2/in1");
lgraph = connectLayers(lgraph,"activation_7_relu","res2c_branch2a");
lgraph = connectLayers(lgraph,"activation_7_relu","add_3/in2");
lgraph = connectLayers(lgraph,"bn2c_branch2c","add_3/in1");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch2a");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch1");
lgraph = connectLayers(lgraph,"bn3a_branch1","add_4/in2");
lgraph = connectLayers(lgraph,"bn3a_branch2c","add_4/in1");
lgraph = connectLayers(lgraph,"activation_13_relu","res3b_branch2a");
lgraph = connectLayers(lgraph,"activation_13_relu","add_5/in2");
lgraph = connectLayers(lgraph,"bn3b_branch2c","add_5/in1");
lgraph = connectLayers(lgraph,"activation_16_relu","res3c_branch2a");
lgraph = connectLayers(lgraph,"activation_16_relu","add_6/in2");
lgraph = connectLayers(lgraph,"bn3c_branch2c","add_6/in1");
lgraph = connectLayers(lgraph,"activation_19_relu","res3d_branch2a");
lgraph = connectLayers(lgraph,"activation_19_relu","add_7/in2");
lgraph = connectLayers(lgraph,"bn3d_branch2c","add_7/in1");
lgraph = connectLayers(lgraph,"activation_22_relu","res4a_branch2a");
lgraph = connectLayers(lgraph,"activation_22_relu","res4a_branch1");
lgraph = connectLayers(lgraph,"bn4a_branch2c","add_8/in1");
lgraph = connectLayers(lgraph,"bn4a_branch1","add_8/in2");
lgraph = connectLayers(lgraph,"activation_25_relu","res4b_branch2a");
lgraph = connectLayers(lgraph,"activation_25_relu","add_9/in2");
lgraph = connectLayers(lgraph,"bn4b_branch2c","add_9/in1");
lgraph = connectLayers(lgraph,"activation_28_relu","res4c_branch2a");
lgraph = connectLayers(lgraph,"activation_28_relu","add_10/in2");
lgraph = connectLayers(lgraph,"bn4c_branch2c","add_10/in1");
lgraph = connectLayers(lgraph,"activation_31_relu","res4d_branch2a");
lgraph = connectLayers(lgraph,"activation_31_relu","add_11/in2");
lgraph = connectLayers(lgraph,"bn4d_branch2c","add_11/in1");
lgraph = connectLayers(lgraph,"activation_34_relu","res4e_branch2a");
lgraph = connectLayers(lgraph,"activation_34_relu","add_12/in2");
lgraph = connectLayers(lgraph,"bn4e_branch2c","add_12/in1");
lgraph = connectLayers(lgraph,"activation_37_relu","res4f_branch2a");
lgraph = connectLayers(lgraph,"activation_37_relu","add_13/in2");
lgraph = connectLayers(lgraph,"bn4f_branch2c","add_13/in1");
lgraph = connectLayers(lgraph,"activation_40_relu","res5a_branch2a");
lgraph = connectLayers(lgraph,"activation_40_relu","res5a_branch1");
lgraph = connectLayers(lgraph,"bn5a_branch1","add_14/in2");
lgraph = connectLayers(lgraph,"bn5a_branch2c","add_14/in1");
lgraph = connectLayers(lgraph,"activation_43_relu","res5b_branch2a");
lgraph = connectLayers(lgraph,"activation_43_relu","add_15/in2");
lgraph = connectLayers(lgraph,"bn5b_branch2c","add_15/in1");
lgraph = connectLayers(lgraph,"activation_46_relu","res5c_branch2a");
lgraph = connectLayers(lgraph,"activation_46_relu","add_16/in2");
lgraph = connectLayers(lgraph,"bn5c_branch2c","add_16/in1");
%% 训练网络
% 使用指定的选项和训练数据对网络进行训练。

[net, traininfo] = trainNetwork(augimdsTrain,lgraph,opts);