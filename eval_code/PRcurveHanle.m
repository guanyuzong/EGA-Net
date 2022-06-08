function PRcurveHanle
%������Ĺ����Ƕ�������������ȡ�Ľ������PR���ߡ�
%by hanlestudy@163.com
clc
clear
close all
%��ȡ���ݿ�
imnames=dir('D:\DPANet-attetion\DPANet-master\rgbd_map3\rgbd_pred139\None\NLPR');  
imnames2=dir('D:\DPANet-attetion\data\RGBD_sal\test\NLPR\mask');  
num=length(imnames);
Precision=zeros(256,num);
Recall=zeros(256,num);
TP=zeros(256,num);
FP=zeros(256,num);
FN=zeros(256,num);
MAES=zeros(num,1);
for j=1:num
    Target=imread(imnames2(j).name);%��ͼ
    Output=imread(imnames(j).name);
    target=rgb2gray(Target);        %��ֵ��ground-truth
    target0=(target)>0;
    for i=0:255
        %��iΪ��ֵ��ֵ��Output
        output0=(Output)>i;
        output1=output0*2;
        TFNP=zeros(256,256);
        x=1;
        TFNP(:,:)=output1(:,:,x)-target0;
        TP(i+1,j)=length(find(TFNP==1));
        FP(i+1,j)=length(find(TFNP==2));
        FN(i+1,j)=length(find(TFNP==-1));
        Precision(i+1,j)=TP(i+1,j)/(TP(i+1,j)+FP(i+1,j));
        Recall(i+1,j)=TP(i+1,j)/(TP(i+1,j)+FN(i+1,j));
    end
    j
end
P=mean(Precision,2);
R=mean(Recall,2);
figure,plot(R,P)  
