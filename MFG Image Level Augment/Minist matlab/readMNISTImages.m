function images = readMNISTImages(filename)
%
%���ܣ���ȡMNIST���ݼ��е�ͼƬ
%Input��
%filename - �ļ�����
%
%Output��
%images - ��ȡ����ͼƬ��28*28*ͼƬ����

FID=fopen(filename,'r');  %fopen()������ĵĺ���,�����ļ�; 'r'�������

%��ȡǰ16���ֽڣ�һ���ֽڰ�λ�����������ע����ѵ����ͼƬΪ����
magic = fread(FID, 1, 'int32', 0, 'ieee-be');            %0 0 8 3 -> 00000000 00000000 00000100 00000011 -> 2051   
numImages = fread(FID, 1, 'int32', 0, 'ieee-be');   %0 0 234 96 -> 60000
numRows = fread(FID, 1, 'int32', 0, 'ieee-be');      %0 0 0 28 -> 28
numCols = fread(FID, 1, 'int32', 0, 'ieee-be');        %0 0 0 28 -> 28

%��ȡ��СΪ28*28��ͼƬ
images = fread(FID, inf, 'unsigned char'); %sizeA  ��������ά����3�ֲ���,Inf��n��[m,n]��Inf ����������������������ļ���ÿһ��Ԫ�ض�Ӧһ��ֵ
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);%������һά�͵ڶ�ά,�൱��ת��;��ΪͼƬ�Ƿ���������Ҫ��תһ��

fclose(FID);

end