%GEN_MAT   generates some test data for ffdiag
%
% Example:
%         gen_mat; [V,CD,stat]=ffdiag(C0);
%
% (c) 2004, Andreas Ziehe, Fraunhofer FIRST.IDA



N=20; % dimensionality of the matrices

K=50;  % number of matrices


sigma=0.0; %noise level 
	   %Note that the algorithm makes use of an
           %approximation which is only valid for SMALL noise levels 
           %try e.g. sigma=0.01; sigma=0.05
	   
A=randn(N); %mixing matrix

L=2*randn(N,K)-1; %generate diagonal elements (eigenvalues)
C0=zeros(N,N,K);  %initialize 3d-array

% Target matrices have to be SYMMETRIC!

P2=(ones(N,N)-eye(N))/sqrt(2)+eye(N)/2;


for k=1:K,
   D=diag(L(:,k)); %build diagonal matrix
   V=randn(N);     %random noise
   SV=(V+V').*P2;  %scaled, symmetrized
   C0(:,:,k)=A*D*A'+sigma*SV; %generate target matrix
end




