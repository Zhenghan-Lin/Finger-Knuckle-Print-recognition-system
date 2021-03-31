function code = dpcode(temp_response, indx)

    d2 = indx + 1;
    if (temp_response(d2+1) > temp_response(d2-1))
        lddp = indx*2;
    else
        lddp = indx*2-1;
    end
    code = lddp;

end