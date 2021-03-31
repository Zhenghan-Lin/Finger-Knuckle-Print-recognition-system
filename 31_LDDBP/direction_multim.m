function [dd_res ss_res] = direction_multim(a, res)

a_length = length(a);
b = zeros(1, a_length+2);
b(2:a_length+1) = a;
b(1) = a(a_length);
b(a_length+2) = a(1);

d_res = [];
s_res = [];
for i=1:a_length
    if (b(i+1)==1)&&(b(i+2)==0)
        d_res = [d_res; i res(i)];
    end
    
    if (b(i+1)==0)&&(b(i+2)==1)
        s_res = [s_res; i res(i)];
    end
end

dd_res = d_res;
ss_res = s_res;

if length(dd_res)>1
    dd_res = sortrows(d_res, -2);
    ss_res = sortrows(s_res, 2);
end

end