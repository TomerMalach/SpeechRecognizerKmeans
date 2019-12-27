function [R] = AutoCorrelationPerColumn(A, p)
    N = size(A, 2);
    % for N frames, calculate the (!)normalized(!) autocorrelation topelitz matrix
    
    R = zeros(p + 1, N);
    
    for k=1:N
        ac = xcorr(A(:, k), p, 'normalized'); 
        R(:, k) = ac(p+1:end);
    end
    
end

