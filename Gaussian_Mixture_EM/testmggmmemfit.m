% Marek Grzes
% University of Waterloo, 2011
% ---

% These are original Gaussians from the gmm_clustering_demo
%mu1 = [1 2];
%sigma1 = [3 .2; .2 2];
%mu2 = [-1 -2];
%sigma2 = [2 0; 0 1];
%X = [mvnrnd(mu1,sigma1,200);mvnrnd(mu2,sigma2,100)];

mu1 = [1 2];
sigma1 = [3 .2; .2 2];
mu2 = [-1 -2];
sigma2 = [2 0; 0 1];
X = [mvnrnd(mu1,sigma1,200);mvnrnd(mu2,sigma2,100)];

% use this for non-debug mode
%gm = mggmmemfit(X, 2, 0, 0, 0);
gm = mggmmemfit(X, 2, 1, 1, 1);

scatter(X(:,1),X(:,2),10,'ko')

hold on
ezcontour(@(x,y)pdf(gm,[x y]),[-8 8],[-8 8]);
hold off
