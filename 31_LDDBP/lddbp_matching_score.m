function score = lddbp_matching_score(vec1, vec2)

    block_num = 64;
    st_i = vec1;
    st_j = vec2;

    st_cha = st_i-st_j;
    st_cha2 = st_cha.*st_cha;
    st_un = st_i+st_j;        
    I = find(st_un ~= 0);
    st_sum = st_cha2(I)./st_un(I);        
    score = sum(st_sum)/block_num; 
        
end
