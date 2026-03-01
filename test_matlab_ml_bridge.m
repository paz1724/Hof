function results = test_matlab_ml_bridge(pythonExe)
% test_matlab_ml_bridge  MATLAB wrapper that exercises matlab_ml_bridge.py
%
% Generates synthetic data, calls the Python bridge for training and
% prediction, and analyses the returned output. Prints a pass/fail
% summary for each check.
%
% Usage:
%   results = test_matlab_ml_bridge();            % auto-detect Python
%   results = test_matlab_ml_bridge("C:\...\python.exe");
%
% Output:
%   results - struct array with fields: name, passed, detail

if nargin < 1
    pythonExe = "";
end

results = struct('name', {}, 'passed', {}, 'detail', {});

% ---- helpers -----------------------------------------------------------
    function add(name, passed, detail)
        r.name = name;
        r.passed = passed;
        r.detail = detail;
        results(end+1) = r; %#ok<AGROW>
    end

% ======================================================================
% 0. Python setup
% ======================================================================
try
    configure_python(pythonExe);
    ensure_python_module_path(fileparts(mfilename("fullpath")));

    % Force single-process sklearn: MATLAB's embedded Python cannot spawn
    % loky worker subprocesses (TerminatedWorkerError with n_jobs=-1).
    % Must set via Python's os.environ (setenv alone is not visible to Python).
    pyrun("import os; os.environ['SKLEARN_N_JOBS'] = '1'");

    bridge = py.importlib.import_module("matlab_ml_bridge");
    py.importlib.reload(bridge);
    add("python_import", true, "matlab_ml_bridge imported OK");
catch ME
    add("python_import", false, ME.message);
    print_summary(results);
    return;
end

% ======================================================================
% 1. Generate synthetic training data
% ======================================================================
rng(42);
nTrain = 200;
nTest  = 80;

[sInTrain, sOutTrain] = generate_data(nTrain);
[sInTest,  sOutTest]  = generate_data(nTest);

XTrain = data_to_X(sInTrain);
yTrain = double(sOutTrain.isFront(:));
XTest  = data_to_X(sInTest);
yTest  = double(sOutTest.isFront(:));

add("data_generation", true, ...
    sprintf("Train: %d samples, %d features | Test: %d samples", ...
            size(XTrain,1), size(XTrain,2), size(XTest,1)));

% ======================================================================
% 2. Train via Python bridge (train_predict_save)
% ======================================================================
trainPayload = struct();
trainPayload.X = XTrain;
trainPayload.y = yTrain;

saveDir = fullfile(tempdir, "matlab_ml_bridge_test");
if isfolder(saveDir)
    rmdir(saveDir, 's');
end

try
    d = struct2pydict(trainPayload);
    res = bridge.train_predict_save(d, saveDir, int32(42));
    add("train_predict_save", true, "Returned without error");
catch ME
    add("train_predict_save", false, ME.message);
    print_summary(results);
    return;
end

% ======================================================================
% 3. Check returned dict keys
% ======================================================================
hasChosen  = logical(py.operator.contains(res, "chosen"));
hasTree    = logical(py.operator.contains(res, "tree"));
hasYProb   = logical(py.operator.contains(res, "y_prob"));
hasMetrics = logical(py.operator.contains(res, "metrics"));

allKeys = hasChosen && hasTree && hasYProb && hasMetrics;
add("result_keys", allKeys, ...
    sprintf("chosen=%d tree=%d y_prob=%d metrics=%d", ...
            hasChosen, hasTree, hasYProb, hasMetrics));

% ======================================================================
% 4. Validate chosen model family and specific model name
% ======================================================================
chosen = string(res{"chosen"});
validFamilies = ["tree", "gnn"];
familyOK = any(chosen == validFamilies);

treeName = string(res{"tree"}{"model"});
validTreeNames = ["rf", "et", "hgb"];
treeNameOK = any(treeName == validTreeNames);

nameOK = familyOK && treeNameOK;
add("chosen_model_name", nameOK, ...
    sprintf("chosen = '%s', tree model = '%s'", chosen, treeName));

% ======================================================================
% 5. Validate training metrics are in [0,1]
% ======================================================================
acc = double(res{"metrics"}{"acc"});
f1  = double(res{"metrics"}{"f1"});
auc = double(res{"metrics"}{"auc"});

metricsOK = acc >= 0 && acc <= 1 && f1 >= 0 && f1 <= 1 && auc >= 0 && auc <= 1;
add("train_metrics_range", metricsOK, ...
    sprintf("acc=%.3f  f1=%.3f  auc=%.3f", acc, f1, auc));

% ======================================================================
% 6. Validate y_prob length matches training N
% ======================================================================
yProbTrain = py_to_double_vector(res{"y_prob"});
lenOK = numel(yProbTrain) == nTrain;
add("y_prob_length", lenOK, ...
    sprintf("Expected %d, got %d", nTrain, numel(yProbTrain)));

% ======================================================================
% 7. Validate y_prob values are in [0,1]
% ======================================================================
probRangeOK = all(yProbTrain >= 0) && all(yProbTrain <= 1);
add("y_prob_range", probRangeOK, ...
    sprintf("min=%.4f  max=%.4f", min(yProbTrain), max(yProbTrain)));

% ======================================================================
% 8. Saved artefacts exist
% ======================================================================
summaryFile = fullfile(saveDir, "summary.json");
summaryExists = isfile(summaryFile);
add("summary_json_exists", summaryExists, summaryFile);

treePath = string(res{"tree"}{"path"});
modelExists = isfile(treePath);
add("model_file_exists", modelExists, char(treePath));

% ======================================================================
% 9. summary.json is valid JSON with expected fields
% ======================================================================
if summaryExists
    try
        txt = fileread(summaryFile);
        summary = jsondecode(txt);
        jsonOK = isfield(summary, 'chosen') && isfield(summary, 'metrics');
        add("summary_json_valid", jsonOK, "Has 'chosen' and 'metrics' fields");
    catch ME
        add("summary_json_valid", false, ME.message);
    end
else
    add("summary_json_valid", false, "File does not exist");
end

% ======================================================================
% 10. Predict on test data (predict_tree_model)
% ======================================================================
try
    pred = bridge.predict_tree_model(char(treePath), py.numpy.array(XTest));
    yProbTest = py_to_double_vector(pred{"y_prob"});
    testPredOK = numel(yProbTest) == nTest;
    add("predict_tree_model", testPredOK, ...
        sprintf("Returned %d probabilities for %d test samples", ...
                numel(yProbTest), nTest));
catch ME
    add("predict_tree_model", false, ME.message);
    yProbTest = [];
end

% ======================================================================
% 11. MATLAB-side test evaluation
% ======================================================================
if ~isempty(yProbTest)
    yPredTest = double(yProbTest >= 0.5);
    testAcc = mean(yPredTest == yTest);
    testF1  = binary_f1(yTest, yPredTest);
    evalOK  = testAcc > 0.5;  % better than random on this easy synthetic data
    add("test_accuracy", evalOK, ...
        sprintf("acc=%.3f  f1=%.3f (threshold: >0.5)", testAcc, testF1));
else
    add("test_accuracy", false, "No test predictions available");
end

% ======================================================================
% 12. Error handling: missing data raises error
% ======================================================================
try
    badPayload = struct2pydict(struct('y', [0;1;0;1]));
    bridge.train_predict_save(badPayload, saveDir, int32(0));
    add("error_missing_X", false, "Expected error was not raised");
catch
    add("error_missing_X", true, "ValueError raised as expected");
end

% ======================================================================
% 13. Error handling: bad model_path raises error
% ======================================================================
try
    bridge.predict_tree_model("", py.numpy.array(XTest));
    add("error_empty_model_path", false, "Expected error was not raised");
catch
    add("error_empty_model_path", true, "ValueError raised as expected");
end

% ======================================================================
% Cleanup & summary
% ======================================================================
if isfolder(saveDir)
    rmdir(saveDir, 's');
end

print_summary(results);

end

% =====================================================================
% Local functions
% =====================================================================

function [sIn, sOut] = generate_data(N)
% Synthetic binary classification data with separable features.
sIn  = struct();
sOut = struct();
idx = randperm(N);
nFront = round(0.5 * N);
idFront = idx(1:nFront);
idBack  = idx(nFront+1:end);

sOut.isFront = false(N,1);
sOut.isFront(idFront) = true;

sIn.LER_dB    = zeros(N,1);
sIn.LER_dB(idFront) = -15 + 4*randn(numel(idFront),1);
sIn.LER_dB(idBack)  = -35 + 4*randn(numel(idBack),1);

sIn.numPeaks = zeros(N,1);
sIn.numPeaks(idFront) = 1 + (rand(numel(idFront),1) > 0.8);
sIn.numPeaks(idBack)  = 2 - (rand(numel(idBack),1)  > 0.8);

sIn.DFErr_deg = zeros(N,1);
sIn.DFErr_deg(idFront) = 2*randn(numel(idFront),1);
sIn.DFErr_deg(idBack)  = 5*randn(numel(idBack),1);

sIn.ENV_dB = zeros(N,1);
sIn.ENV_dB(idFront) = -12 + 5*randn(numel(idFront),1);
sIn.ENV_dB(idBack)  = -8  + 7*randn(numel(idBack),1);

sIn.SNR_dB = zeros(N,1);
sIn.SNR_dB(idFront) = 16 + 4*randn(numel(idFront),1);
sIn.SNR_dB(idBack)  =  8 + 5*randn(numel(idBack),1);

sIn.Freq_MHz = 95 + 10*randn(N,1);
end

function X = data_to_X(sIn)
X = [sIn.LER_dB(:), sIn.numPeaks(:), sIn.DFErr_deg(:), ...
     sIn.ENV_dB(:), sIn.SNR_dB(:), sIn.Freq_MHz(:)];
end

function f1 = binary_f1(yTrue, yPred)
yTrue = logical(yTrue(:));
yPred = logical(yPred(:));
tp  = sum(yTrue & yPred);
fp  = sum(~yTrue & yPred);
fn  = sum(yTrue & ~yPred);
den = 2*tp + fp + fn;
if den == 0
    f1 = 0.0;
else
    f1 = 2*tp / den;
end
end

function configure_python(pythonExe)
if strlength(string(pythonExe)) == 0
    return;
end
pe = pyenv;
if pe.Status == "Loaded"
    if ~strcmpi(string(pe.Executable), string(pythonExe))
        error("Python already loaded from: %s. Restart MATLAB to change.", string(pe.Executable));
    end
else
    pyenv("Version", char(pythonExe));
end
end

function ensure_python_module_path(moduleDir)
sys = py.importlib.import_module("sys");
pyPath = cellfun(@string, cell(sys.path), 'UniformOutput', true);
if ~any(pyPath == string(moduleDir))
    sys.path.insert(int32(0), char(moduleDir));
end
end

function d = struct2pydict(s)
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
    out = py.numpy.array(v);
elseif ischar(v) || isstring(v)
    out = char(v);
elseif iscell(v)
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
arr = py.numpy.asarray(pyObj);
x = double(arr).';
x = x(:);
end

function print_summary(results)
fprintf("\n========== MATLAB ML Bridge Test Summary ==========\n");
nPass = 0;
nFail = 0;
for i = 1:numel(results)
    r = results(i);
    if r.passed
        tag = "PASS";
        nPass = nPass + 1;
    else
        tag = "FAIL";
        nFail = nFail + 1;
    end
    fprintf("  [%s] %s : %s\n", tag, r.name, r.detail);
end
fprintf("---------------------------------------------------\n");
fprintf("  Total: %d | Passed: %d | Failed: %d\n", nPass+nFail, nPass, nFail);
fprintf("===================================================\n");
end
