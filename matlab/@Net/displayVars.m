function displayVars(net, varargin)
%DISPLAYVARS
%   Simple table of information on each var, and corresponding derivative.
%
%   NET.DISPLAYVARS(VARS)
%   Uses the given variables list, rather than NET.VARS. This is useful
%   for debugging inside calls to NET.EVAL, where NET.VARS is empty and
%   a local variable VARS is used instead, for performance reasons.
%
%   NET.DISPLAYVARS(___, 'OPT', VAL, ...) accepts the following options:
%
%   `showRange`:: `true`
%      If set to true, shows columns with the minimum and maximum for each
%      variable.
%
%   `showMemory`:: `true`
%      If set to true, shows columns with the memory taken by each
%      variable.
%
%   `showLinks`:: `true` if not running Matlab in a terminal
%      If set to true, shows hyperlinks that print the syntax to access
%      the value of each variable (e.g. 'net.vars{INDEX}').


% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


if ~isempty(varargin) && iscell(varargin{1})
  vars = varargin{1} ;
  varargin(1) = [] ;
else
  vars = net.vars ;
end
opts.showRange = true ;
opts.showMemory = true ;
%opts.showRF = true ;
opts.showLinks = usejava('desktop') ;
opts = vl_argparse(opts, varargin) ;

if isempty(vars)
  error(sprintf(['NET.VARS is empty.\n' ...
    'NOTE: Net.eval() is executing. For performance, it holds all of the\n' ...
    'network''s variables in a local variable (called ''vars''). To display\n' ...
    'them, first navigate to the scope of Net.eval() with dbup/dbdown.\n\n'])) ;  %#ok<SPERR>
end

info = net.getVarsInfo() ;
assert(numel(info) == numel(vars)) ;


if opts.showLinks
  if ~isempty(inputname(1))
    netname = inputname(1) ;
  else  % unnamed Net object
    netname = 'net' ;
  end
  if nargin >= 2 && ~isempty(inputname(2))  % a variables list was given
    varname = inputname(2) ;
  else  % only a Net object was given
    varname = [netname '.vars'] ;
  end
end
keyboard

% print information for each variable
funcs = cell(numel(vars), 1) ;
values = cell(numel(vars), 1) ;
flags = cell(numel(vars), 1) ;
mins = NaN(numel(vars), 1) ;
maxs = NaN(numel(vars), 1) ;
mem = zeros(numel(vars), 1) ;
rf = cell(numel(vars), 1) ;
for i = 1:numel(vars)
  % function of each layer, as a string
  if strcmp(info(i).type, 'layer')
    funcs{i} = func2str(net.forward(info(i).index).func) ;
    if info(i).outputArgPos > 1
      funcs{i} = sprintf('%s (output #%i)', funcs{i}, info(i).outputArgPos) ;
    end
  
    if opts.showLinks
      % link to get to the originating layer (fwd/bwd structs)
      fwd = sprintf('%s.forward(%i)', netname, info(i).index) ;
      bwd = sprintf('%s.backward(%i)', netname, numel(net.forward) - info(i).index + 1) ;
      
      funcs{i} = sprintf(['<a href="matlab:if exist(''%s'',''var''),disp(''%s ='');disp(%s);' ...
        'disp(''%s ='');disp(%s);else,disp(''%s, %s'');end">%s</a>'], ...
        netname, fwd, fwd, bwd, bwd, fwd, bwd, funcs{i}) ;
    end
  else
    % other var types like 'param' or 'input' will be displayed as such
    funcs{i} = info(i).type ;
  end
  
  % size and underlying type (e.g. single 50x3x2)
  v = vars{i} ;
  if isa(v, 'gpuArray')
    str = classUnderlying(v) ;
  else
    str = class(v) ;
  end
  str = [str ' ' sprintf('%ix', size(v))] ;  %#ok<*AGROW>
  values{i} = str(1:end-1) ;  % remove extraneous 'x' at the end of the var size
  
  if opts.showLinks
    % link to show variable's value
    title = info(i).name ;
    if mod(i, 2) == 0
      title = [title ' derivative'] ;
    end
    values{i} = sprintf('<a href="matlab:figure(''Name'',''%s'');vl_tshow(%s{%d});colorbar">%s</a>', ...
      title, varname, i, values{i}) ;
  end
  
  % flags (GPU, NaN, Inf)
  str = '' ;
  if isa(v, 'gpuArray')
    str = [str 'GPU '] ;
  end
  if any(isnan(v(:)))
    str = [str 'NaN '] ;
  end
  if any(isinf(v(:)))
    str = [str 'Inf '] ;
  end
  if isempty(str)
    flags{i} = ' ' ;  % no flags, must still have a space
  else
    flags{i} = str(1:end-1) ;  % delete extra space at the end
  end
  
  % min and max
  if opts.showRange
    if isnumeric(v) && ~isempty(v)
      mins(i) = gather(min(v(:))) ;
      maxs(i) = gather(max(v(:))) ;
    end
  end

  % receptive fields
  %TODO: dilated convolutions
  if opts.showRF
    %if strcmp(info(i).type, 'layer')
      %l = net.forward(info(i).index) ;
      %if isequal(l.func, @vl_nnconv)
        %sz = size(vars{l.inputVars(2)}) 
        %rf{i} = sz(1:2) 
      %end
    %end
       %%switch func
         %%case @vl_nnconv
           %%net.forward(idx).inputVars(2)
         %%otherwise
           %%keyboard
        %%end
       %%net.forward(idx).func
    %else
      %rf{i} = [] ;
    %end
  end
  
  % memory
  if opts.showMemory
    type = class(v) ;
    if strcmp(type, 'gpuArray')
      type = classUnderlying(v) ;
    end
    switch type
    case 'double'
      bytes = 8 ;
    case 'single'
      bytes = 4 ;
    case 'logical'
      bytes = 1 ;
    otherwise  % e.g. 'uint32', get the '32'. otherwise bytes will be NaN.
      bytes = str2double(regexprep(type, '[a-z]', '')) / 8 ;
    end
    if ~isreal(v)
      bytes = 2 * bytes ;
    end
    mem(i) = bytes * numel(v) ;
  end
end

if opts.showRange  % convert to string, filling NaNs with spaces
  minStr = num2str(mins, '%.2g') ;
  minStr(isnan(mins),:) = ' ' ;
  minStr = num2cell(minStr, 2) ;
  
  maxStr = num2str(maxs, '%.2g') ;
  maxStr(isnan(maxs),:) = ' ' ;
  maxStr = num2cell(maxStr, 2) ;
end

%if opts.showRF
  %%rfStr = cellfun(@num2str, rf, 'Uni', 0) ;
  %%rfStr = cellfun(@(x) formatRF(x), rf, 'Uni', 0) ;
  %%, rfStr = cellfun(@(x) sprintf('[%d,%d]', x(:), rf, 'Uni', 0) ;
%end

if opts.showMemory
  suffixes = {'B ', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB'} ;
  place = floor(log(mem) / log(1024)) ;  % 0-based index into 'suffixes'
  place(mem == 0) = 0 ;  % 0 bytes needs special handling
  num = mem ./ (1024 .^ place) ;
  
  memStr = num2str(num, '%.0f')  ;
  memStr(:,end+1) = ' ' ;
  memStr = [memStr, char(suffixes{max(1, place + 1)})] ;  % concatenate number and suffix
  
  memStr(isnan(mem),:) = ' ' ;  % leave invalid values blank
  memStr = num2cell(memStr, 2) ;
end


idx = arrayfun(@(i) {num2str(i)}, (1:2:numel(info)-1)') ;

% now print out the info as a table
table = [{'Idx', 'Function', 'Name'};
  idx, funcs(1:2:end-1), {info(1:2:end-1).name}'] ;

% repeat same set of columns for value and der (size/class/flags/min/max)
headers = {'Value', 'Derivative'} ;

for i = 1:2
  idx = i : 2 : numel(values) - 2 + i ;  % odd or even elements, respectively
  
  table = [table, [{headers{i}, 'Flags'}; values(idx), flags(idx)]] ;
  
  if opts.showRange
    table = [table, [{'Min', 'Max'}; minStr(idx), maxStr(idx)]] ;
  end

  % display receptive fields only for forward direction
  %if opts.showRF && (i == 1) 
    %table = [table, [{'RF'}; rfStr(idx)]] ;
  %end
  
  if opts.showMemory
    table = [table, [{'Memory'}; memStr(idx)]] ;
  end
end

% ensure all cells contain strings
table(cellfun('isempty', table)) = {''};

% align column contents
for i = 1:size(table,2)
  table(:,i) = leftAlign(table(:,i)) ;
end

% add spaces between columns
t = cell(size(table,1), size(table,2) * 2 - 1) ;
t(:,1:2:end) = table ;
t(:,2:2:end) = {'  '} ;
table = t ;

% concatenate and display
for i = 1:size(table, 1)
  disp([table{i,:}]) ;
end
fprintf('\n') ;

end

function str = leftAlign(str)
  % aligns cell array of strings to the left, as a column, ignoring links
  strNoLinks = regexprep(str, '<[^>]*>', '') ;  % remove links
  lengths = cellfun('length', strNoLinks) ;
  numBlanks = max(lengths) - lengths ;
  str = cellfun(@(s, n) {[s blanks(n)]}, str, num2cell(numBlanks)) ;
end

function str = formatRF(x)
  if ~isempty(x)
    keyboard
  else
    str = '' ;
end

