clear
rng default; % For reproducibility
prwaitbar off;

%% Information
%Area.
%Perimeter.
%Compactness
%Length of kernel.
%Width of kernel.
%Asymmetry coefficient.
%Length of kernel groove.
%Class (1, 2, 3).

%Purpose of analysis
%To investigate if it is possible to successfully classify the three varieties of wheat with an accuracy higher than 90% and which method is best? 

%Discus
%The mahalanobis distance classifier is the one with the highest accuarcy
%compared to the other classifiers.

%% Data Load
seedsData = fopen('data.txt','r');
seeds = textscan(seedsData,'%s%s%s%s%s%s%s%s%s%s%s', 'delimiter',',');
fclose(seedsData);

%% Feature Vector

%Area, Perimeter, Length of kernel, Class
FV = [seeds{1,1} seeds{1,2} seeds{1,3} seeds{1,4} seeds{1,5} seeds{1,6} seeds{1,7} seeds{1,8} seeds{1,9} seeds{1,10}];
FV_labs = seeds{1,11};
[I, J] = size(FV);

%FV_1 = [FV(1:70,1) FV(1:70,2) FV(1:70,3)];
%FV_2 = [FV(71:140,1) FV(71:140,2) FV(71:140,3)];
%FV_3 = [FV(141:210,1) FV(141:210,2) FV(141:210,3)];


%scatter plot Area against Perimeter
%figure(1);
%scatter(FV_1(:,1),FV_1(:,2),25, [1 0 0], 'filled');
%hold on
%scatter(FV_2(:,1),FV_2(:,2), 25, [0 1 0], 'filled');
%scatter(FV_3(:,1),FV_3(:,2), 25, [0 0 1], 'filled');
%xlabel('Area');
%ylabel('Perimeter');
%%hold off
%Scatter plot area against length of kernel
%figure(2);
%scatter(FV_1(:,1),FV_1(:,3),25, [1 0 0], 'filled');
%hold on
%scatter(FV_2(:,1),FV_2(:,3), 25, [0 1 0], 'filled');
%scatter(FV_3(:,1),FV_3(:,3), 25, [0 0 1], 'filled');
%xlabel('Area');
%ylabel('Length of kernel');
%title('Graph showing three types of seeds');

%% Pearson Correlation
SeedDataCorr = [seeds{1,1} seeds{1,2} seeds{1,3} seeds{1,4} seeds{1,5} seeds{1,6} seeds{1,7} seeds{1,8} seeds{1,9} seeds{1,10}];
%SeedDataCorr = [seeds{1,5} seeds{1,6} seeds{1,7}]
[Seed_I, Seed_J] = size(SeedDataCorr);
Seed_PC = corr(SeedDataCorr)

%% Eigen Decomposition 
%Sample Mean Vector
Seed_mean = mean(SeedDataCorr);

%Centerize data column/featurewise
Seed_X_C = SeedDataCorr-ones(I,1)*Seed_mean;

%Sample covariance 
Seed_Cov = cov(SeedDataCorr);

%Eigenvalue decomposition for symmetric matrices
[Seed_V, Seed_D] = eig(Seed_Cov);

%Pick eigenvalues from diagonal
Seed_Eigval = diag(Seed_D);

%Sort eigenvectors and eigenvalues in descending order
[Seed_Eigval, Seed_idx] = sort(Seed_Eigval,'descend'); 
Seed_V = Seed_V(:,Seed_idx);

%Plot of eigen vectors
hold off; figure(3);
%plot(Seed_V(:,Seed_J));
plot(Seed_Eigval)
xlabel('X'); ylabel('Y'); title('Eigenvalues');

f = figure ; linmod ={ 'r','b:','g -. ','k*-'}
hold on ;
for J =1:4
    plot ( Seed_V(:,Seed_J) , linmod { J })
end

%set(gca ,'XTick ' ,[1 2 3 4 5 6 7] , ' XTickLabel ' ,{'Sepal L','Sepal W', 'Petal L','Petal W'})
%legend ('1. Eigvec ','2. Eigvec ','3. Eigvec ','4. Eigvec ')


%Calculate and plot the cumulative sum of eigenvalue percentage
Seed_cs = cumsum(Seed_Eigval) / sum(Seed_Eigval); 

%Plot of cumulative sum of eigenvalue percentage.
%figure(4);
%plot(Seed_cs);
%title('Cumulative sum of eigenvalue percentage'); ylabel('Eigenvalue Percentage'); xlabel('Eigenvalue');

%Sample Mean Vector
FV_mean = mean(FV);

%Centerize data column/featurewise
FV_X_C = FV-ones(I,1)*FV_mean;

%Sample covariance 
FV_Cov = cov(FV);

%Eigenvalue decomposition for symmetric matrices
[V, D] = eig(FV_Cov);

%Pick eigenvalues from diagonal
FV_Eigval = diag(D);

%Sort eigenvectors and eigenvalues in descending order
[FV_Eigval, idx] = sort(FV_Eigval,'descend'); 
V = V(:,idx);

% Reduction of dimension from 3 to 2
scores = Seed_X_C * Seed_V(:,1:2);

% Create a pr data set with all the data
FV_prdataset = prdataset(SeedDataCorr, FV_labs);

% Create a pr data set with three features
FV_prdataset = prdataset(FV, FV_labs);
                                                                                                                                                                                                
% Create a pr data set with scores
FV_prdataset_scores = prdataset(scores, FV_labs);

% Reconstruct data
Xr = scores*V(:,1:3)+ones(I,1)*FV_mean;

% Split data into two datasets, train and test.
[FV_train, FV_test] = gendat(FV_prdataset, 0.7);

%% K-means Clustering 
%K-means
%eva = evalclusters(FV,'kmeans','CalinskiHarabasz','KList',[1:20])

idx = prkmeans(FV_prdataset, 3);
figure(5); hold off;
scatter(FV_prdataset(:,1), FV_prdataset(:,2), [],idx);
xlabel('X'); ylabel('Y'); title('Kmeans clustering');

%% Minimum Distance Classifier / Nearest Mean Scaled Classifier(Linear)
mdc_w = nmsc(FV_train);% Train Classifier
mdc_w_scores= nmsc(FV_prdataset_scores);
mdc_d = FV_test*mdc_w;                                         % Classify test set   
mdc_cm = confmat(mdc_d);                                       % Confusion Matrix
mdc_acc = sum(diag(mdc_cm))/sum(sum(mdc_cm));                  % Calculate accuracy

hold off; figure(6);
scatterd(FV_test);
hold on;
plotm(mdc_w);
title('Minimum Distance Classifier');

hold off; figure(7);
scatterd(FV_test);
hold on;
plotc(mdc_w_scores);
title('Minimum Distance Classifier');

%% k-nearest neighbour classifier
knnc_u = knnc([]);
%knnc_classifier = FV_train*knnc_u;                       % train the classifier
[knnc_classifier, k_knnc] = knnc(FV_train)
knnc_ds = FV_test*knnc_classifier;                       % apply classifier on the test set
knnc_ds_plot = FV_prdataset_scores*knnc_u;
knnc_cm = confmat(knnc_ds)    ;                                  % Confusion Matrix
knnc_acc = sum(diag(knnc_cm))/sum(sum(knnc_cm));                   % Calculate accuracy

hold off; figure(8);
scatterd(FV_train);                 
axis equal; hold on;
plotc(knnc_ds_plot); 
title('K-nearst Neighbor Classifier');

%% Quadratic Bayes Normal Classifier (Quadratic)
QBN_w = qdc(FV_train);                                         % Train Classifier
QBN_w_plot = qdc(FV_prdataset_scores);
QBN_d = FV_test*QBN_w;                                         % Classify training set                                           % Cisplay confusion matrix
QBN_cm = confmat(QBN_d) ;                                      % Confusion Matrix
QBN_acc = sum(diag(QBN_cm))/sum(sum(QBN_cm));                   % Calculate accuracy

figure(11) ; scatterd (FV_train) ; hold on ; plotm (QBN_w_plot) ; title('Quadratic Bayes Normal Classifier');
figure(12) ; scatterd (FV_test) ; hold on ; plotc (QBN_w_plot) ; title('Quadratic Bayes Normal Classifier');

%% Mahalanobis Distance and Minimum Mahalanobis Classifier (Linear discriminant)
MD_w = ldc(FV_train);
MD_w_plot = ldc(FV_prdataset_scores);
MD_d = FV_test*MD_w;
MD_cm = confmat(MD_d);                                      % Confusion Matrix
MD_acc = sum(diag(MD_cm))/sum(sum(MD_cm));                   % Calculate accuracy

% Plots
figure(13) ; scatterd (FV_train) ; hold on ; plotm (MD_w_plot) ; title('Magalanobis Distance and minimum mahalanobis Classifier');
figure(14) ; scatterd (FV_test) ; hold on ; plotc (MD_w_plot) ; title('Magalanobis Distance and minimum mahalanobis Classifier');

%% Naive Bayes Classifier (probabilistic classifier)

NBC_w = naivebc(FV_train);
NBC_w_plot = naivebc(FV_prdataset_scores);
NBC_d = FV_test*NBC_w;
NBC_cm = confmat(NBC_d);                                      % Confusion Matrix
NBC_acc = sum(diag(NBC_cm))/sum(sum(NBC_cm));                   % Calculate accuracy

figure(15);
scatterd(FV_prdataset_scores);
plotc(NBC_w_plot);
title('Naive Bayes Classifier');
 
 %% Neural Network 
 
NN_w = rnnc(FV_train,5);
NN_w_plot = rnnc(FV_prdataset_scores,5);
NN_d = FV_test*NN_w;
NN_cm = confmat(NN_d);                                      % Confusion Matrix
NN_acc = sum(diag(NN_cm))/sum(sum(NN_cm));                   % Calculate accuracy

figure(16);
scatterd(FV_test);
plotc(NN_w_plot);
title('Neural Network Classifier');

%% Print Results

% Cross Validation using LDC
e1_LDC = crossval(FV_train,ldc,4,1);
e5_LDC = crossval(FV_train,ldc,4,5);
e25_LDC = crossval(FV_train,ldc,4,25);
edps_LDC = crossval(FV_train,ldc,4,'DPS');

% Cross Validation using nmsc
e1_nmsc = crossval(FV_train,nmsc,4,1);
e5_nmsc = crossval(FV_train,nmsc,4,5);
e25_nmsc = crossval(FV_train,nmsc,4,25);
edps_nmsc = crossval(FV_train,nmsc,4,'DPS');

% Cross Validation using qdc
e1_qdc = crossval(FV_train,qdc,4,1);
e5_qdc = crossval(FV_train,qdc,4,5);
e25_qdc = crossval(FV_train,qdc,4,25);
edps_qdc = crossval(FV_train,qdc,4,'DPS');

% Cross Validation using naivebc
e1_naivebc = crossval(FV_train,naivebc,4,1);
e5_naivebc = crossval(FV_train,naivebc,4,5);
e25_naivebc = crossval(FV_train,naivebc,4,25);
edps_naivebc = crossval(FV_train,naivebc,4,'DPS');

% Cross Validation using knnc
e1_knnc = crossval(FV_train,knnc,4,1);
e5_knnc = crossval(FV_train,knnc,4,5);
e25_knnc = crossval(FV_train,knnc,4,25);
edps_knnc = crossval(FV_train,knnc,4,'DPS');

% Cross Validation using rnnc
e1_rnnc = crossval(FV_train,rnnc,4,1);
e5_rnnc = crossval(FV_train,rnnc,4,5);
e25_rnnc = crossval(FV_train,rnnc,4,25);
edps_rnnc = crossval(FV_train,rnnc,4,'DPS');


disp([e1_LDC,e5_LDC,e25_LDC,edps_LDC]);
%(e1_LDC+e5_LDC+e25_LDC+edps_LDC)/4

disp([e1_nmsc,e5_nmsc,e25_nmsc,edps_nmsc]);
%(e1_nmsc+e5_nmsc+e25_nmsc+edps_nmsc)/4

disp([e1_qdc,e5_qdc,e25_qdc,edps_qdc]);
%(e1_qdc+e5_qdc+e25_qdc+edps_qdc)/4

disp([e1_naivebc,e5_naivebc,e25_naivebc,edps_naivebc]);
%(e1_naivebc+e5_naivebc+e25_naivebc+edps_naivebc)/4

disp([e1_knnc,e5_knnc,e25_knnc,edps_knnc]);
%(e1_knnc+e5_knnc+e25_knnc+edps_knnc)/4

disp([e1_rnnc,e5_rnnc,e25_rnnc,edps_rnnc]);
%(e1_rnnc+e5_rnnc+e25_rnnc+edps_rnnc)/4

disp([mdc_acc, knnc_acc, QBN_acc, MD_acc, NBC_acc, NN_acc])
