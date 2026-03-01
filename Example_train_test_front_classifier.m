function out = Example_train_test_front_classifier(pythonExe, seed)
% Example MATLAB wrapper:
% 1) Generates synthetic measurement fields in sIn
% 2) Generates binary target sOut.isFront
% 3) Trains in Python (decision-tree family via matlab_ml_bridge.py)
% 4) Generates independent test set, predicts in Python, evaluates in MATLAB
%
% Inputs:
%   pythonExe (optional) - full path to python.exe used by MATLAB
%   seed (optional) - random seed (default 42)
%
% Output:
%   out struct with train/test metrics and predictions

if nargin < 1
    pythonExe = "";
end
if nargin < 2 || isempty(seed)
    seed = 42;
end

rng(seed);

% ------------------------ Generate train/test data ------------------------
nTrain = 1200;
nTest = 500;
[sInTrain, sOutTrain] = generate_measurement_data(nTrain);
[sInTest, sOutTest] = generate_measurement_data(nTest);

XTrain = measurements_to_X(sInTrain);
yTrain = double(sOutTrain.isFront(:));
XTest = measurements_to_X(sInTest);
yTest = double(sOutTest.isFront(:));

% ------------------------ Train in Python ------------------------
trainPayload = struct();
trainPayload.X = XTrain;
trainPayload.y = yTrain;

res = Train_and_predict(trainPayload, pythonExe, seed);
chosen = string(res{"chosen"});

% For this wrapper, we do explicit test inference with the saved tree model.
if ~isKey(res, "tree")
    error("Tree model result was not returned. This wrapper expects tabular tree training.");
end
treePath = string(res{"tree"}{"path"});

bridge = py.importlib.import_module("matlab_ml_bridge");
py.importlib.reload(bridge);
pred = bridge.predict_tree_model(char(treePath), py.numpy.array(XTest));
yProbTest = py_to_double_vector(pred{"y_prob"});
yPredTest = double(yProbTest >= 0.5);

% ------------------------ MATLAB evaluation ------------------------
accTest = mean(yPredTest == yTest);
f1Test = binary_f1(yTest, yPredTest);

fprintf("Chosen model (selection metric): %s\n", chosen);
fprintf("Test metrics in MATLAB: acc=%.3f | f1=%.3f\n", accTest, f1Test);

out = struct();
out.train = struct();
out.train.chosen = chosen;
out.train.acc_select = double(res{"metrics"}{"acc"});
out.train.f1_select = double(res{"metrics"}{"f1"});
out.train.auc_select = double(res{"metrics"}{"auc"});

out.test = struct();
out.test.acc = accTest;
out.test.f1 = f1Test;
out.test.y_prob = yProbTest;
out.test.y_pred = yPredTest;
out.test.y_true = yTest;
end

function [sIn, sOut] = generate_measurement_data(N)
% Generate synthetic measurements with physically-plausible ranges.
% Inputs requested by user:
%   sIn.LER_dB, sIn.numPeaks, sIn.DFErr_deg, sIn.ENV_dB, sIn.SNR_dB, sIn.Freq_MHz
% Output:
%   sOut.isFront in {false,true}

% Input fields
sIn = struct();
idx = randperm(N);
nFront = round(0.5 * N);
idFront = idx(1:nFront);
idBack = idx(nFront+1:end);

% Ground-truth output (two sets by design).
sOut = struct();
sOut.isFront = false(N,1);
sOut.isFront(idFront) = true;   % Set 1
sOut.isFront(idBack) = false;   % Set 2

% Set 1 (isFront=1): higher LER_dB, numPeaks=1 with p=0.8
% Set 2 (isFront=0): lower LER_dB, numPeaks=2 with p=0.8
sIn.LER_dB = zeros(N,1);
sIn.LER_dB(idFront) = -15 + 4 * randn(numel(idFront),1);  % higher
sIn.LER_dB(idBack) = -35 + 4 * randn(numel(idBack),1);    % lower

sIn.numPeaks = zeros(N,1);
sIn.numPeaks(idFront) = 1 + (rand(numel(idFront),1) > 0.8); % 1 with p=0.8 else 2
sIn.numPeaks(idBack) = 2 - (rand(numel(idBack),1) > 0.8);   % 2 with p=0.8 else 1

% Other measurement inputs (class-dependent but overlapping).
sIn.DFErr_deg = zeros(N,1);
sIn.DFErr_deg(idFront) = 2.0 * randn(numel(idFront),1);
sIn.DFErr_deg(idBack) = 5.0 * randn(numel(idBack),1);

sIn.ENV_dB = zeros(N,1);
sIn.ENV_dB(idFront) = -12 + 5 * randn(numel(idFront),1);
sIn.ENV_dB(idBack) = -8 + 7 * randn(numel(idBack),1);

sIn.SNR_dB = zeros(N,1);
sIn.SNR_dB(idFront) = 16 + 4 * randn(numel(idFront),1);
sIn.SNR_dB(idBack) = 8 + 5 * randn(numel(idBack),1);

sIn.Freq_MHz = 95 + 10 * randn(N,1);
end

function X = measurements_to_X(sIn)
X = [ ...
    double(sIn.LER_dB(:)), ...
    double(sIn.numPeaks(:)), ...
    double(sIn.DFErr_deg(:)), ...
    double(sIn.ENV_dB(:)), ...
    double(sIn.SNR_dB(:)), ...
    double(sIn.Freq_MHz(:)) ...
];
end

function f1 = binary_f1(yTrue, yPred)
yTrue = logical(yTrue(:));
yPred = logical(yPred(:));
tp = sum(yTrue & yPred);
fp = sum(~yTrue & yPred);
fn = sum(yTrue & ~yPred);
den = 2 * tp + fp + fn;
if den == 0
    f1 = 0.0;
else
    f1 = 2 * tp / den;
end
end

function x = py_to_double_vector(pyObj)
arr = py.numpy.asarray(pyObj);
x = double(arr).';
x = x(:);
end
