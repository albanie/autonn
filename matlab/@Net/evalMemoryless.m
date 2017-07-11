function vars = evalMemoryless(net, vars, precious)
%EVALMEMORYLESS test-mode network evaluation without intermediate storage
%   VARS = EVALMEMORYLESS(NET, VARS, PRECIOUS) computes a forward pass of 
%   `NET` in which intermediate activations are cleared after they are no 
%   longer needed. This adds a small runtime overhead, but can be useful 
%   for enabling larger batch sizes. 
%
%   `VARS` is a cell array containing a copy of the network variables 
%   `PRECIOUS` is a cell array of variable names to be preserved during
%    execution. If empty, the final output variable is saved by default.
%
% Copyright (C) 2017 Samuel Albanie and Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  forward = net.forward ;
  deps = zeros(size(vars)) ; % compute var dependency graph
  if isempty(precious)
    deps(forward(end).outputVar) = 1 ; % final output preserved
  else
    deps(net.getVarIndex(precious)) = 1 ; % mark variables as "precious"
  end
  for k = 1:numel(forward)
    layer = forward(k) ; 
    deps(layer.inputVars) = deps(layer.inputVars) + 1 ;
  end
  for k = 1:numel(forward)
    layer = forward(k) ;
    args = layer.args ;
    args(layer.inputArgPos) = vars(layer.inputVars) ;
    out = cell(1, max(layer.outputArgPos)) ;
    [out{:}] = layer.func(args{:}) ;

    vars(layer.outputVar) = out(layer.outputArgPos);
    deps(layer.inputVars) = deps(layer.inputVars) - 1 ; 
    drop = find(deps == 0) ; vars(drop) = {[]} ; % clear up
    deps(drop) = deps(drop) - 1 ; % mark as cleared
  end
end
