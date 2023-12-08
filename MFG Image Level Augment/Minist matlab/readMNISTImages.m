function images = readMNISTImages(filename)
%
%功能：读取MNIST数据集中的图片
%Input：
%filename - 文件名称
%
%Output：
%images - 读取到的图片（28*28*图片数）

FID=fopen(filename,'r');  %fopen()是最核心的函数,导入文件; 'r'代表读入

%读取前16个字节（一个字节八位），（这儿的注释以训练集图片为例）
magic = fread(FID, 1, 'int32', 0, 'ieee-be');            %0 0 8 3 -> 00000000 00000000 00000100 00000011 -> 2051   
numImages = fread(FID, 1, 'int32', 0, 'ieee-be');   %0 0 234 96 -> 60000
numRows = fread(FID, 1, 'int32', 0, 'ieee-be');      %0 0 0 28 -> 28
numCols = fread(FID, 1, 'int32', 0, 'ieee-be');        %0 0 0 28 -> 28

%读取大小为28*28的图片
images = fread(FID, inf, 'unsigned char'); %sizeA  输出数组的维度有3种参数,Inf、n、[m,n]，Inf 代表输出数据是列向量，文件中每一个元素对应一个值
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);%交换第一维和第二维,相当于转置;因为图片是反的所以需要翻转一下

fclose(FID);

end