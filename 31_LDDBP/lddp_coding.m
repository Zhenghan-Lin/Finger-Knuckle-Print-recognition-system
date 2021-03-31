function [fir_pat sec_pat] = lddp_coding(img2)

load('gaborfilter.mat');

img = double(255-img2);
% get img's size
[row col]= size(img);
% the length of code equals the quantity of filters
code_length = length(filters);
% initialize an array to store the convolutional results
d_response = zeros(row, col, code_length+2);

% do the convolutional process, 
% and results are stored in the boxes from 2-13
filter_size = code_length;
for k=1:filter_size
    d_response(:,:,k+1) = (conv2(img, filters{k}, 'same'));
end
% box1 equals box13 | box14 equals box2
d_response(:,:,1) = d_response(:,:,code_length+1);
d_response(:,:,code_length+2) = d_response(:,:,2);

% encode lddbp code
mul_code = zeros(row, col, code_length);
for k=1:code_length
    mul_code(:,:,k) = d_response(:,:,k+1)>d_response(:,:,k);
end

% do some preparation for comparison of code
temp_code = zeros(code_length,1);
temp_response = zeros(code_length+2,1);
code1 = zeros(row, col);
code2 = zeros(row, col);

% comparison begin
for i=1:row
    for j=1:col
        temp_code(:) = mul_code(i,j,:);
        temp_response(:) = d_response(i,j,:);
        
        [d_res, s_res] = direction_multim(temp_code,temp_response(2:13));
        
        res_size = size(d_res,1);
        if (res_size == 0)
            continue;
        else
            d1 = d_res(1,1);
            tc = dpcode(temp_response, d1);
            s1 = s_res(1,1);
            code1(i,j) = (tc-1)*11 + mod(12+s1-d1,12);
            
            if (res_size >= 2)
                fd2 = d_res(2,1);
                tcc = dpcode(temp_response, fd2);
                sd2 = s_res(2,1);
                code2(i,j) = (tcc-1)*11 + mod(sd2+12-fd2,12);
            end
            
        end
    end
end
% return Lm(fir_pat) and Ls(sec_pat)
fir_pat = code1;
sec_pat = code2;

end