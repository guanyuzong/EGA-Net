%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Harris角点检测算法 Matlab code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc ;tic;
 
ori_im = imread('opencv10.jpg');     % 读取图像
 
if(size(ori_im,3)==3)
    ori_im = rgb2gray(uint8(ori_im));  %转为灰度图像
end
 
% fx = [5 0 -5;8 0 -8;5 0 -5];          % 高斯函数一阶微分，x方向(用于改进的Harris角点提取算法)
fx = [-2 -1 0 1 2];                 % x方向梯度算子(用于Harris角点提取算法)
Ix = filter2(fx,ori_im);              % x方向滤波
% fy = [5 8 5;0 0 0;-5 -8 -5];          % 高斯函数一阶微分，y方向(用于改进的Harris角点提取算法)
fy = [-2;-1;0;1;2];                 % y方向梯度算子(用于Harris角点提取算法)
Iy = filter2(fy,ori_im);              % y方向滤波
Ix2 = Ix.^2;
Iy2 = Iy.^2;
Ixy = Ix.*Iy;
clear Ix;
clear Iy;
 
h= fspecial('gaussian',[7 7],2);      % 产生7*7的高斯窗函数，sigma=2
 
Ix2 = filter2(h,Ix2);
Iy2 = filter2(h,Iy2);
Ixy = filter2(h,Ixy);
 
height = size(ori_im,1);
width = size(ori_im,2);
result = zeros(height,width);         % 纪录角点位置，角点处值为1
 
R = zeros(height,width);
for i = 1:height
    for j = 1:width
        M = [Ix2(i,j) Ixy(i,j);Ixy(i,j) Iy2(i,j)];             % auto correlation matrix
        R(i,j) = det(M)-0.06*(trace(M))^2;   
    end
end
cnt = 0;
for i = 2:height-1
    for j = 2:width-1
        % 进行非极大抑制，窗口大小3*3
        if  R(i,j) > R(i-1,j-1) && R(i,j) > R(i-1,j) && R(i,j) > R(i-1,j+1) && R(i,j) > R(i,j-1) && R(i,j) > R(i,j+1) && R(i,j) > R(i+1,j-1) && R(i,j) > R(i+1,j) && R(i,j) > R(i+1,j+1)
            result(i,j) = 1;
            cnt = cnt+1;
        end
    end
end
Rsort=zeros(cnt,1);
[posr, posc] = find(result == 1);
for i=1:cnt
    Rsort(i)=R(posr(i),posc(i));
end
[Rsort,ix]=sort(Rsort,1);
Rsort=flipud(Rsort);
ix=flipud(ix);
ps=100;
posr2=zeros(ps,1);
posc2=zeros(ps,1);
for i=1:ps
    posr2(i)=posr(ix(i));
    posc2(i)=posc(ix(i));
end
   
imshow(ori_im);
hold on;
plot(posc2,posr2,'g+');
 
toc;