function b = AtRadon6(patch, eigenVecs)
%% For a particular Patch, Get the Alphas
b = eigenVecs' * patch;