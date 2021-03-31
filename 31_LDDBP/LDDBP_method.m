clc;
clear all;

img_size = 128;
row_col = img_size*img_size;
row = img_size;
col = img_size;

img1 = imread('1.bmp');
img2 = imread('2.bmp');


[fir_code_1(:,:) fir_code_2(:,:)]= lddp_coding(img1);
[sec_code_1(:,:) sec_code_2(:,:)]= lddp_coding(img2);

%=================================================================
% block size is 16, there are 8*8 blocks(bk_num) in total
bk_size = 16; bk_size2 = bk_size*bk_size;
bk_num = floor(img_size/bk_size);

binCenter = 1:264;
secCenter = 1:264;

firvector1 = [];
firvector2 = [];
% the range is 1-8, generate first img's lddbp descriptor
for r = 1:bk_num
    for c = 1:bk_num
% initialize the index of start and end
% begin at certain block's start and finish at the end of that block
        r_start = (r-1)*bk_size+1;
        r_end = r*bk_size;

        if r_end > (img_size)
            r_end = (img_size);
        end

        c_start = (c-1)*bk_size+1;
        c_end = c*bk_size;
        if c_end > (img_size)
            c_end = (img_size);
        end

        fir_temp1 = fir_code_1(r_start:r_end, c_start:c_end);
%         transfer rows to columns and eliminate 0
        fir_temp1(find(fir_temp1==0)) = [];
%         compute Lm histogram
        firvector1 = [firvector1 hist(fir_temp1(:), binCenter)/bk_size2];

        fir_temp2 = fir_code_2(r_start:r_end, c_start:c_end);
%         transfer rows to columns and eliminate 0
        fir_temp2(find(fir_temp2==0)) = [];
%         compute Ls histogram
        firvector2 = [firvector2 hist(fir_temp2(:), secCenter)/bk_size2];

    end
end
% concatenate the Lm and Ls descriptor
lddbp_block_code_1 = [firvector1'; firvector2'];

% the range is 1-8, generate second img's lddbp descriptor
firvector1 = [];
firvector2 = [];
for r = 1:bk_num
    for c = 1:bk_num
        r_start = (r-1)*bk_size+1;
        r_end = r*bk_size;

        if r_end > (img_size)
            r_end = (img_size);
        end

        c_start = (c-1)*bk_size+1;
        c_end = c*bk_size;
        if c_end > (img_size)
            c_end = (img_size);
        end

        fir_temp1 = sec_code_1(r_start:r_end, c_start:c_end);
        fir_temp1(find(fir_temp1==0)) = [];
        firvector1 = [firvector1 hist(fir_temp1(:), binCenter)/bk_size2];

        fir_temp2 = sec_code_2(r_start:r_end, c_start:c_end);
        fir_temp2(find(fir_temp2==0)) = [];
        firvector2 = [firvector2 hist(fir_temp2(:), secCenter)/bk_size2];

    end
end
    
lddbp_block_code_2 = [firvector1'; firvector2'];

matching_score = lddbp_matching_score(lddbp_block_code_1, lddbp_block_code_2);

matching_score



