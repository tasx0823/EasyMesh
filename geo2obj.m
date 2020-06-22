%geometry to mesh (.off)
clear,clc
%data_path = 'I:\ICCV2019\sp1\test_sample'
exp = 'haha';
%data_path = 'I:\CVPR_rebuttal\new_data\sp\car\sp7_1'
%data_path = ['I:\ICCV2019\sp3\',cate];
%data_path = ['I:\ICCV2019\sp3\',cate,'5'];
%data_path = ['I:\ICCV2019\results\all_models\',cate,'\10_2_',cate,'_test_samples']
%data_path = 'I:\ICCV2019\results\all_models\lamp\lamp_test_samples'
% data_path = ['I:\ICCV2019\results\selection\',cate,'\10_2_airplane_test_samples']
% data_path = ['E:\pycode\geoimage\data\',cate,'\',cate,'_test_geoimg'];
% data_path = 'E:\pycode\geoimage\data\car\new_model_6.17\test_sample_1';
%data_path = ['I:\ICCV2019\results\ablation\',cate,'\',exp,'_table_test_samples']

cate = 'car';
data_path = [cate,'.mat'];

data = load(data_path);
data = data.geoimgs;


for idxx = 1:size(data,1)
%for idxx = 1:2
    sample = squeeze(data(idxx,:,:,:));
    
    cut_pixel = 1;
    sample = my_interpolation(sample,cut_pixel,'linear');
    
   
    %target_path = ['I:\ICCV2019\results\all_models\',cate,'\',cate,'_',num2str(idxx),'.obj'];
    %target_path = 'I:\ICCV2019\results\surfnet\car_9.obj';
    %target_path = ['I:\ICCV2019\results\ablation\',cate,'\z',cate,num2str(idxx),'_',exp,'.obj'];
    
    target_path = ['sample\',cate,'_',num2str(idxx),'.obj'];
    
    %----------------reduce the number of point cloud------------------%
    sample = imresize(sample,[32,32]);
    sample = imresize(sample,[48,48]);
    %sample = imresize(sample,[128,128]);
    %------------------------------------------------------------------%

    [m,n,c] = size(sample);
    triangles = zeros((m-1)*(n-1)*2,4);
    cnt = 0;
    edge_dist = 0;
    edge_cnt = 0;
    for i = 1:m-1
        for j = 1:n-1
            %low trianle
            cnt = cnt + 1;
            triangles(cnt,1) = 3;
            triangles(cnt,2) = (i-1)*(n)+j;     %(i,j)
            triangles(cnt,3) = (i)*(n)+j;       %(i+1,j)
            triangles(cnt,4) = (i)*(n)+j+1;     %(i+1,j+1)

            %up triangle
            cnt = cnt + 1;
            triangles(cnt,1) = 3;
            triangles(cnt,2) = (i-1)*(n)+j;     %(i,j)
            triangles(cnt,3) = (i-1)*(n)+j+1;   %(i,j+1)
            triangles(cnt,4) = (i)*(n)+j+1;     %(i+1,j+1)
        end
    end
    
   %---------------------four edges--------------------------%

    for i = 1:m/2-1
        %low triangle
        cnt = cnt + 1;
        triangles(cnt,1) = 3;
        triangles(cnt,2) = m/2-(i-1);
        triangles(cnt,3) = m/2-(i-1)-1;
        triangles(cnt,4) = m/2+(i-1)+1;
        
        %up triangle
        cnt = cnt + 1;
        triangles(cnt,1) = 3;
        triangles(cnt,2) = m/2-(i-1)-1;  
        triangles(cnt,3) = m/2+(i-1)+1;  
        triangles(cnt,4) = m/2+(i-1)+2; 
    end
    

     for i = 1:m/2-1
        %low triangle
        cnt = cnt + 1;
        triangles(cnt,1) = 3;
        triangles(cnt,2) = (n-1)*m+m/2-(i-1);
        triangles(cnt,3) = (n-1)*m+m/2-(i-1)-1;
        triangles(cnt,4) = (n-1)*m+m/2+(i-1)+1;
        
        %up triangle
        cnt = cnt + 1;
        triangles(cnt,1) = 3;
        triangles(cnt,2) = (n-1)*m+m/2-(i-1)-1;  
        triangles(cnt,3) = (n-1)*m+m/2+(i-1)+1;  
        triangles(cnt,4) = (n-1)*m+m/2+(i-1)+2; 
     end
    

     for i = 1:m/2-1
        %low triangle
        cnt = cnt + 1;
        triangles(cnt,1) = 3;
        triangles(cnt,2) = (m/2-(i-1)-1)*m+1;
        triangles(cnt,3) = (m/2-(i-1)-1-1)*m+1;
        triangles(cnt,4) = (m/2+(i-1)+1-1)*m+1;
        
        %up triangle
        cnt = cnt + 1;
        triangles(cnt,1) = 3;
        triangles(cnt,2) = (m/2-(i-1)-1-1)*m+1;  
        triangles(cnt,3) = (m/2+(i-1)+1-1)*m+1;  
        triangles(cnt,4) = (m/2+(i-1)+2-1)*m+1; 
     end
    

     for i = 1:m/2-1
        %low triangle
        cnt = cnt + 1;
        triangles(cnt,1) = 3;
        triangles(cnt,2) = (m/2-(i-1))*m;
        triangles(cnt,3) = (m/2-(i-1)-1)*m;
        triangles(cnt,4) = (m/2+(i-1)+1)*m;
        
        %up triangle
        cnt = cnt + 1;
        triangles(cnt,1) = 3;
        triangles(cnt,2) = (m/2-(i-1)-1)*m;  
        triangles(cnt,3) = (m/2+(i-1)+1)*m;  
        triangles(cnt,4) = (m/2+(i-1)+2)*m; 
     end
     
    
    if cut_pixel>0
       % up triangle
       cnt = cnt + 1;
       triangles(cnt,1) = 3;
       triangles(cnt,2) = 1;  
       triangles(cnt,3) = m;  
       triangles(cnt,4) = (m) * (m-1) + 1;
          
       %low triangle
       cnt = cnt + 1;
       triangles(cnt,1) = 3;
       triangles(cnt,2) = m;  
       triangles(cnt,3) = (m) * (m-1) + 1;
       triangles(cnt,4) = m*m;
       
    end
    %-------------------------------------------------------%
    
    triangles = triangles(1:cnt,:);


    gim = reshape(permute(sample,[2,1,3]),m*n,3);
    gim = gim - mean(gim);

    %----------write into files---------------%
    fid = fopen(target_path,'w');
    for i = 1:size(gim,1)
        fprintf(fid,'%s %f %f %f\r\n','v',gim(i,:));
    end
    for i = 1:size(triangles,1)
        fprintf(fid,'%s %d %d %d\r\n','f',triangles(i,2:4));
    end
    fclose(fid);
    %-----------------------------------------%
end