load('dataKomp.mat');

rng default  % for reproducibility
k = fix(length(dataKomp)*0.5); % datanýn yüzde 30 u için çapraz doðrulama yapýlýyor
input = dataKomp(:,1:6); % SILT-KUM-ÇAKIL-ÖZGÜLAÐIRLIK-LÝKÝTLÝMÝT-PLASTÝCLÝMÝT-OPTÝMUMSUÝÇERÝÐÝ-MAX.KUR.BR.AÐ.
output1 = dataKomp(:,7);
output2 = dataKomp(:,8);

%% MLSSVR
output = dataKomp(:,7:8);
%[gamma, lambda, p, MSE] = GridMLSSVR(input, output, 10);
gamma = 15;lambda=10; p=3;
dataindex = 1:numel(input(:,1));
Y = zeros(37,2);
for i=1:10
    test = randsample(numel(input(:,1)),k);    
    trn = dataindex(~ismember(dataindex,test));
    [alpha, b] = MLSSVRTrain(input(trn,:), output(trn,:), gamma, lambda, p);
    [predicted_label, total_squared_error, squared_correlation_coefficient] = MLSSVRPredict(input(test,:), output(test,:), input(trn,:), alpha, b, lambda, p);
    X = output(test,:);
    Y = Y + predicted_label;
    r =(corrcoef(X,predicted_label));R(i,1) =r(1,2);
    absD = abs(X-predicted_label);
    mse(i,:) = sum(absD)/numel(X);
end
RAvg = mean(R);
Y = bsxfun(@rdivide,Y,10);
figure
bar(1:10,R,'b');
xlabel('Num. of iteration')
ylabel('R-value')
title('Pearson correlation for predicted output space')
figure
bar(1:10,mse);
legend('Mse.(OMC)','Mse.(MDD)')
xlabel('Num. of iteration')
ylabel('Error rate')
title('Mean Squared Error for predicted output space')
figure;
plotregression(X,Y,'Regression')
legend('Multivariate regression plot')
%% SVM Regression

mdlSVM1 = fitrsvm(input,output1,'KernelFunction','rbf','KernelScale','auto','Standardize',true);
mdlSVM2 = fitrsvm(input,output2,'KernelFunction','rbf','KernelScale','auto','Standardize',true);
% 
mdlSVM1 = fitrsvm(input,output1,'KernelFunction','linear','KernelScale','auto','Standardize',true);
mdlSVM2 = fitrsvm(input,output2,'KernelFunction','linear','KernelScale','auto','Standardize',true);

mdlSVM1 = fitrsvm(input,output1,'KernelFunction','polynomial','KernelScale','auto','Standardize',true);
mdlSVM2 = fitrsvm(input,output2,'KernelFunction','polynomial','KernelScale','auto','Standardize',true);

mdl1 = mdlSVM1;
mdl2 = mdlSVM2;
CVMdl1 = crossval(mdl1,'KFold',k); %Cross-validated support vector machine regression model
CVMdl2 = crossval(mdl2,'KFold',k); %Cross-validated support vector machine regression model
loss1 = kfoldLoss(CVMdl1);
loss2 = kfoldLoss(CVMdl2);
yfit1 = kfoldPredict(CVMdl1);
yfit2 = kfoldPredict(CVMdl2);
 for i = 1:10
    idx1 = randsample(numel(yfit1),k);
    idx2 = randsample(numel(yfit2),k);
%     t1 = table(output1(idx1),yfit1(idx1),'VariableNames',{'TrueLabels','PredictedLabels'});
%     t2 = table(output2(idx2),yfit2(idx2),'VariableNames',{'TrueLabels','PredictedLabels'});
%     corrplot(t1);
%     corrplot(t2);
    X1 = output1(idx1);
    X2 = output2(idx2);
    Y1 = yfit1(idx1);
    Y2 = yfit2(idx2);
    r1 =(corrcoef(X1,Y1));R1(i,1) = r1(1,2);
    r2 =(corrcoef(X2,Y2));R2(i,1) = r2(1,2);
    absD1 = abs(X1-Y1);
    absD2 = abs(X2-Y2);
    mse1(i,1) = sum(absD1(:))/numel(X1);
    mse2(i,1) = sum(absD2(:))/numel(X2);    
 end
R1Avg = mean(R1);
R2Avg = mean(R2);
MSE1Avg = mean(mse1(i,1));
MSE2Avg = mean(mse2(i,1));
figure
R1 = round(R1,2);
R2 = round(R2,2);
R = ([R1 R2]);
bar(R);
text(1:length(R1),R1,num2str(R1),'vert','bottom','horiz','center');
text(1:length(R2),R2,num2str(R2),'vert','bottom','horiz','center'); 
box off
legend('R-OMC','R-MDD')
xlabel('Num. of iteration');
ylabel('R-value')
title('Pearson correlation for predicted output space')
figure
MSE1 = round(mse1,2);
MSE2 = round(mse2,2);
MSE = ([MSE1 MSE2]);
bar(MSE);
text(1:length(MSE1),MSE1,num2str(MSE1),'vert','bottom','horiz','center'); 
text(1:length(MSE2),MSE2,num2str(MSE2),'vert','bottom','horiz','center');
box off
legend('Mse.(OMC)','Mse.(MDD)')
xlabel('Num. of iteration')
ylabel('Error rate')
title('Mean Squared Error for predicted output space')
X = [X1,X2];
Y = [Y1,Y2];
figure;
plotregression(X1,Y1,'Regression')
legend('SVR-R Value')
figure;
plotregression(X2,Y2,'Regression')
legend('SVR-R Value')
figure
plotregression(X,Y,'Regression')
legend('SVR-R Value')

%% Decision Tree

idxNaN = isnan(output(:,1) + input(:,1));
X = input(~idxNaN,:);
Y = output(~idxNaN,:);
n = numel(X(:,1));

rng(1) % For reproducibility
idxTrn = false(n,1);
idxTrn(randsample(n,round(0.5*n))) = true; % Training set logical indices
idxVal = idxTrn == false;                  % Validation set logical indices

Mdl1 = fitrtree(X(idxTrn,:),Y(idxTrn,1),'CrossVal','on','PredictorNames',{'Silt','Sand','Gravel','Gs','WL','WP'});
Mdl2 = fitrtree(X(idxTrn,:),Y(idxTrn,2),'CrossVal','on','PredictorNames',{'Silt','Sand','Gravel','Gs','WL','WP'});

mseDefault1 = kfoldLoss(Mdl1);
mseDefault2 = kfoldLoss(Mdl2);

% view(Mdl1.Trained{1},'Mode','graph')
% view(Mdl2.Trained{1},'Mode','graph')

numBranches = @(x)sum(x.IsBranch);
mdlDefaultNumSplits1 = cellfun(numBranches, Mdl1.Trained);
mdlDefaultNumSplits2 = cellfun(numBranches, Mdl2.Trained);

% figure;
% histogram(mdlDefaultNumSplits1)

% figure;
% histogram(mdlDefaultNumSplits2)

m1 = max(Mdl1.Trained{1}.PruneList);
m2 = max(Mdl2.Trained{1}.PruneList);
pruneLevels1 = 0:2:m1; % Pruning levels to consider
pruneLevels2 = 0:2:m2; % Pruning levels to consider
z1 = numel(pruneLevels1);
z2 = numel(pruneLevels2);

Yfit1 = predict(Mdl1.Trained{1},X(idxVal,:),'SubTrees',pruneLevels1);
Yfit2 = predict(Mdl2.Trained{1},X(idxVal,:),'SubTrees',pruneLevels2);

sortDat1 = sortrows([X(idxVal,:) Y(idxVal,1) Yfit1],1); % Sort all data with respect to X
sortDat2 = sortrows([X(idxVal,:) Y(idxVal,2) Yfit2],1); % Sort all data with respect to X
a1 = sortDat1(:,7:end);
a2 = sortDat2(:,7:end);

mserr1 = zeros(size(a1));
mserr1(:,1) = a1(:,1);
for i = 2:size(mserr1,2)
    for j = 1:size(mserr1,1)
        mserr1(j,i) = immse(a1(j,1),a1(j,i));
    end
end

mserr2 = zeros(size(a2));
mserr2(:,1) = a2(:,1);
for i = 2:size(mserr2,2)
    for j = 1:size(mserr2,1)
        mserr2(j,i) = immse(a2(j,1),a2(j,i));
    end
end
figure(1);
plot(sortDat1(:,1),sortDat1(:,7),'*');
hold on;
plot(repmat(sortDat1(:,1),1,size(Yfit1,2)),sortDat1(:,8:end));
lev = cellstr(num2str((pruneLevels1)','Level %d OMC'));
legend(['Observed OMC'; lev])
title 'Out-of-Sample Predictions'
xlabel 'Input';
ylabel 'Optimum Water Content (OMC)';
h = findobj(gcf);
axis tight;
set(h(4:end),'LineWidth',3) % Widen all lines

figure(2);
plot(sortDat2(:,1),sortDat2(:,7),'*');
hold on;
plot(repmat(sortDat2(:,1),1,size(Yfit2,2)),sortDat2(:,8:end));
lev = cellstr(num2str((pruneLevels2)','Level %d MDD'));
legend(['Observed MDD'; lev])
title 'Out-of-Sample Predictions'
xlabel 'Input';
ylabel 'Max.Dry Unit Weight (MDD)';
h = findobj(gcf);
axis tight;
set(h(4:end),'LineWidth',3) % Widen all lines
X1 = a1(:,1);
Y1 = a1(:,5);
X2 = a2(:,1);
Y2 = a2(:,3);
X = [X1 X2];
Y = [Y1 Y2];
figure(3)
plotregression(X1,Y1)
figure(4)
plotregression(X2,Y2)
figure(5)
plotregression(X,Y)
%% Linear Regression for Six Predcitor
 % https://www.mathworks.com/matlabcentral/answers/352044-how-to-use-the-regress-function-for-more-than-2-predictors
X = [ones(size(dataKomp(:,1))) dataKomp(:,2) dataKomp(:,3) dataKomp(:,4) dataKomp(:,5) dataKomp(:,6)];
[b0,bint0,r0,rint0,stats0] = regress(output1,X);
[b1,bint1,r1,rint1,stats1] = regress(output2,X);
YFIT1 = b0(1)*dataKomp(:,1) + b0(2)*dataKomp(:,2) + b0(3)*dataKomp(:,2) + b0(4)*dataKomp(:,4) + b0(5)*dataKomp(:,5) + b0(6)*dataKomp(:,6);
YFIT2 = b1(1)*dataKomp(:,1) + b1(2)*dataKomp(:,2) + b1(3)*dataKomp(:,2) + b1(4)*dataKomp(:,4) + b1(5)*dataKomp(:,5) + b1(6)*dataKomp(:,6);
figure
plotregression(output1,YFIT1,'Regression');
figure
plotregression(output2,YFIT2,'Regression');
%% Linear Regression for Two Predictors
figure;
x11 = dataKomp(:,2); % Kum
x12 = dataKomp(:,3);   % Çakýl
y11 = dataKomp(:,8); % Max Kur. Br. Aðýrlýk
X = [ones(size(x11)) x11 x12 x11.*x12];
[b,bint,r,rint,stats] = regress(y11,X);

figure
scatter3(x11,x12,y11,'filled')
hold on

x1fit = min(x11):5:max(x11);
x2fit = min(x12):5:max(x12);
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT + b(4)*X1FIT.*X2FIT;

mesh(X1FIT,X2FIT,YFIT)
xlabel('Sand')
ylabel('Gravel')
zlabel('Max Dry Unit Weight')
view(50,10)

% meshing optimum water
x21 = dataKomp(:,5); % LL
x22 = dataKomp(:,6);   % PL
y12 = dataKomp(:,8); % Optimum su
X = [ones(size(x21)) x21 x22 x21.*x22];
b = regress(y12,X);

figure
scatter3(x21,x22,y12,'filled')
hold on

x1fit = min(x21):2:max(x21);
x2fit = min(x22):2:max(x22);
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT + b(4)*X1FIT.*X2FIT;

mesh(X1FIT,X2FIT,YFIT)
xlabel('Liqid Limit')
ylabel('Plastic Limit')
zlabel('Optimum Water')
view(50,10)