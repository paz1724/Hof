function [res] = Train_and_predict(sIn, pythonExe, seed)
% Train in Python and return predictions/metrics to MATLAB.
% - sIn: MATLAB struct with fields:
%     tabular mode: sIn.X [N,F], sIn.y [N,1]
%     graph mode:   sIn.node_feats [N,V,F], sIn.edge_index [2,E] or [E,2], sIn.y [N,1]
% - pythonExe (optional): absolute path to python.exe
% - seed (optional): random seed (default 0)

if nargin < 1 || isempty(sIn)
    % Example input if none is provided.
    sIn.X = randn(200,5);       % [N,F]
    sIn.y = randi([0 1],200,1); % [N,1]
end
if nargin < 2
    pythonExe = "";
end
if nargin < 3 || isempty(seed)
    seed = 0;
end

configure_python(pythonExe);
ensure_python_module_path(fileparts(mfilename("fullpath")));

% Force single-process sklearn: MATLAB's embedded Python cannot spawn
% loky worker subprocesses (TerminatedWorkerError with n_jobs=-1).
pyrun("import os; os.environ['SKLEARN_N_JOBS'] = '1'");

d = struct2pydict(sIn);
bridge = py.importlib.import_module("matlab_ml_bridge");
py.importlib.reload(bridge);

saveDir = fullfile(fileparts(mfilename("fullpath")), "py_models");
res = bridge.train_predict_save(d, saveDir, int32(seed));

% Pull outputs back to MATLAB
chosen = string(res{"chosen"});
y_prob = py_to_double_vector(res{"y_prob"});
acc = double(res{"metrics"}{"acc"});
f1  = double(res{"metrics"}{"f1"});
auc = double(res{"metrics"}{"auc"});

fprintf("Chosen model: %s | acc=%.3f | f1=%.3f | auc=%.3f\n", chosen, acc, f1, auc);

% Make MATLAB-side fields easy to consume.
res_matlab = struct();
res_matlab.chosen = chosen;
res_matlab.y_prob = y_prob;
res_matlab.acc = acc;
res_matlab.f1 = f1;
res_matlab.auc = auc;
res{"matlab"} = struct2pydict(res_matlab);
end

function configure_python(pythonExe)
if strlength(string(pythonExe)) == 0
    return;
end

pe = pyenv;
if pe.Status == "Loaded"
    if ~strcmpi(string(pe.Executable), string(pythonExe))
        error(["MATLAB Python is already loaded from: " + string(pe.Executable) + ...
               ". Restart MATLAB and call Train_and_predict with the desired pythonExe first."]);
    end
else
    pyenv("Version", char(pythonExe));
end
end

function ensure_python_module_path(moduleDir)
sys = py.importlib.import_module("sys");
pyPath = cellfun(@string, cell(sys.path), "UniformOutput", true);
if ~any(pyPath == string(moduleDir))
    sys.path.insert(int32(0), char(moduleDir));
end
end

function d = struct2pydict(s)
% Convert MATLAB struct to py.dict recursively (supports numeric, char, cell, struct)
    keys = fieldnames(s);
    d = py.dict;
    for i = 1:numel(keys)
        k = keys{i};
        v = s.(k);
        d{k} = matlab2py(v);
    end
end

function out = matlab2py(v)
    if isstruct(v)
        out = struct2pydict(v);
    elseif isnumeric(v) || islogical(v)
        out = py.numpy.array(v);  % requires numpy installed in that Python
    elseif ischar(v)
        out = py.str(v);
    elseif isstring(v)
        if isscalar(v)
            out = py.str(char(v));
        else
            % String array → Python list of strings
            L = py.list;
            for i = 1:numel(v)
                L.append(py.str(char(v(i))));
            end
            out = L;
        end
    elseif iscell(v)
        % Convert cell to python list
        L = py.list;
        for i = 1:numel(v)
            L.append(matlab2py(v{i}));
        end
        out = L;
    else
        error("Unsupported type: %s", class(v));
    end
end

function x = py_to_double_vector(pyObj)
% Robust conversion from Python list/ndarray to MATLAB double column vector.
arr = py.numpy.asarray(pyObj);
x = double(arr).';
x = x(:);
end


