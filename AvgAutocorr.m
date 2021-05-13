function [ac] = AvgAutocorr(X, lags)
[N,~] = size(X);

AllACs = zeros(N,2*lags+1);
for ni=1:N
    AllACs(ni,:) = xcorr(X(ni,:)',lags,'unbiased');
end
ac = mean(AllACs);

end