function precision = fit_knn(In_Data,In_Flag,k)
% In_data: 输入特征值
% In_flag: 输入特征值对应的标签
% k:输入临近K值
% 输入特征和对应标签,程序自动分出训练和测试数据得到测试准确率.
% date:2017.12.12
feature = In_Data;

stdr=std(feature);                %求各变量的标准差；  
[n,m]=size(feature);               %矩阵的行与列  
sddata=feature./stdr(ones(n,1),:);         %标准化变换  
[p,princ,egenvalue]=princomp(sddata);  %主成分分析  
p=p(:,1:10);                          %输出前10主成分系数；  
sc=princ(:,1:10);                       %前10主成分量；  
egenvalue;                              %相关系数矩阵的特征值，即各主成分所占比例；  
per=100*egenvalue/sum(egenvalue);       %各个主成分所占百分比；

MappedData = mapminmax(sc, 0, 1);   %特征归一化
% MappedData = (sc-min(min(sc)))/(max(max(sc))-min(min(sc)));

Data = MappedData;
% Data(:,11) = cell2mat(In_Flag);
Data(:,11) = In_Flag;
%%-------------------------------------------main---------------------------------
len_Data = length(Data);
Tag = 0;
Pool = [];
Data_test = [];
for i=1:len_Data
    if Tag==3
        Data_test = [Data_test;Data(i,:)];  % 分出1/4测试，其余为Pool
        Tag = 0;
    else
        Pool = [Pool;Data(i,:)];
        Tag = Tag+1;
    end
end

len_DaTest = length(Data_test);
K = k;
id = [];
for i=1:len_DaTest
    idx=knnsearch(Data_test(i,1:10),Pool(:,1:10),K);       % KNN得出预测结果
%     idx = knnsearch(Pool(:,1:10),Data_test(i,1:10),'dist','euclidean','k',K);
    id = [id;idx];
end

[m,n] = size(id);
id_temp = zeros(m,n);
for i=1:m
    for j=1:n
        id_temp(i,j) = Pool(id(i,j),11);      %结果转换成标签
    end
end
[p,q] = size(id_temp);
for i=1:p
    ord = 1:max(id_temp(i,:));
%     ord = 0:9;
    nums = histc(id_temp(i,:),ord);
    [max_num,max_id] = max(nums);
    max_element = ord(max_id);
    id_temp(i,q+1) = max_element;
end
counts = sum(Data_test(:,11)==id_temp(:,q+1));  %对比预测对的数据
precision = counts/p;
disp(['precision=',num2str(precision)]);
end
