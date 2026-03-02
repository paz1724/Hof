function run_all_models_and_plot(pythonExe, seed)
% Train all 8 tabular models, then produce:
%   1. Loss vs epoch plot   (all 8 models — torch as curves, sklearn as horizontal lines)
%   2. Accuracy vs epoch plot (all 8 models)
%   3. Confusion matrices    (all 8 models)
%   4. Save epoch data to .mat
%   5. Save all figures to Outputs/
%
% Usage:
%   run_all_models_and_plot          % default Python, seed=42
%   run_all_models_and_plot("", 42)

if nargin < 1, pythonExe = ""; end
if nargin < 2 || isempty(seed), seed = 42; end

rng(seed);
outDir = fullfile(fileparts(mfilename("fullpath")), "Outputs");
if ~exist(outDir, "dir"), mkdir(outDir); end

% ======================== Generate data ========================
N = 1200;
[sInTrain, sOutTrain] = generate_measurement_data(N);
XTrain = measurements_to_X(sInTrain);
yTrain = double(sOutTrain.isFront(:));

nTest = 500;
[sInTest, sOutTest] = generate_measurement_data(nTest);
XTest = measurements_to_X(sInTest);
yTest = double(sOutTest.isFront(:));

% ======================== Train all 7 models ========================
allModels = ["random_forest", "extra_trees", "hist_gradient_boosting", ...
             "bagged_trees", "svm", "cnn", "transformer", "rl"];
trainPayload = struct('X', XTrain, 'y', yTrain, 'models', allModels);
fprintf("Training all 8 models...\n");
res = Train_and_predict(trainPayload, pythonExe, seed);

% ======================== Load summary.json for epoch data ========================
summaryPath = fullfile(fileparts(mfilename("fullpath")), "py_models", "summary.json");
fid = fopen(summaryPath, 'r');
raw = fread(fid, '*char')';
fclose(fid);
summary = jsondecode(raw);

% ======================== Collect per-model predictions ========================
sklearnKeys = ["random_forest", "extra_trees", "hist_gradient_boosting", "bagged_trees", "svm"];
torchKeys   = ["cnn", "transformer", "rl"];

modelNames = {};
yProbTrain = {};
yProbTest  = {};

bridge = py.importlib.import_module("matlab_ml_bridge");
py.importlib.reload(bridge);

for i = 1:numel(sklearnKeys)
    mkey = sklearnKeys(i);
    fprintf("Retraining %s individually for per-model predictions...\n", mkey);
    p = struct('X', XTrain, 'y', yTrain, 'models', mkey);
    r = Train_and_predict(p, pythonExe, seed);
    treePath = string(r{"tree"}{"path"});
    % Train predictions
    yProbTrain{end+1} = py_to_double_vector(r{"tree"}{"y_prob"});
    % Test predictions
    pred = bridge.predict_tree_model(char(treePath), py.numpy.array(XTest));
    yProbTest{end+1} = py_to_double_vector(pred{"y_prob"});
    modelNames{end+1} = char(mkey);
end

% Torch models — use the all-models result (they have individual entries)
for i = 1:numel(torchKeys)
    mkey = torchKeys(i);
    if isfield(summary, char(mkey)) && isfield(summary.(char(mkey)), 'y_prob')
        yProbTrain{end+1} = summary.(char(mkey)).y_prob(:);
        % For test: re-run prediction via subprocess if needed
        fprintf("Retraining %s for test predictions...\n", mkey);
        pTest = struct('X', [XTrain; XTest], 'y', [yTrain; yTest], 'models', mkey);
        rTest = Train_and_predict(pTest, pythonExe, seed);
        allProb = py_to_double_vector(rTest{char(mkey)}{"y_prob"});
        yProbTest{end+1} = allProb(N+1:end);
    else
        yProbTrain{end+1} = NaN(N,1);
        yProbTest{end+1}  = NaN(nTest,1);
    end
    modelNames{end+1} = char(mkey);
end

% ======================== Epoch history (torch models only) ========================
epochData = struct();
for i = 1:numel(torchKeys)
    mkey = char(torchKeys(i));
    if isfield(summary, mkey) && isfield(summary.(mkey), 'train_info') ...
       && isfield(summary.(mkey).train_info, 'epoch_history')
        eh = summary.(mkey).train_info.epoch_history;
        epochData.(mkey).train_loss = eh.train_loss(:);
        epochData.(mkey).train_acc  = eh.train_acc(:);
        epochData.(mkey).train_f1   = eh.train_f1(:);
        epochData.(mkey).val_loss   = eh.val_loss(:);
        epochData.(mkey).val_acc    = eh.val_acc(:);
        epochData.(mkey).val_f1     = eh.val_f1(:);
    end
end

% ======================== Compute sklearn final metrics ========================
% For sklearn models, compute BCE loss and accuracy as single values
sklearnMetrics = struct();
for i = 1:numel(sklearnKeys)
    mkey = char(sklearnKeys(i));
    p_tr = yProbTrain{i};
    p_te = yProbTest{i};
    sklearnMetrics.(mkey).train_loss = bce_loss(yTrain, p_tr);
    sklearnMetrics.(mkey).train_acc  = mean(double(p_tr >= 0.5) == yTrain);
    sklearnMetrics.(mkey).train_f1   = binary_f1(yTrain, double(p_tr >= 0.5));
    sklearnMetrics.(mkey).val_loss   = bce_loss(yTest, p_te);
    sklearnMetrics.(mkey).val_acc    = mean(double(p_te >= 0.5) == yTest);
    sklearnMetrics.(mkey).val_f1     = binary_f1(yTest, double(p_te >= 0.5));
end

% Save epoch data to .mat
save(fullfile(outDir, "epoch_history.mat"), "epochData", "sklearnMetrics");
fprintf("Saved epoch_history.mat\n");

% ======================== Consistent colors for all 8 models ========================
allModelKeys = [sklearnKeys, torchKeys];
nAllModels = numel(allModelKeys);
cmap = lines(nAllModels);

% ======================== Plot 1: Loss vs Epoch (all 8 models) ========================
fig1 = figure('Name', 'Loss vs Epoch', 'Position', [100 100 900 400]);
maxEpoch = 0;
for i = 1:numel(torchKeys)
    mkey = char(torchKeys(i));
    if isfield(epochData, mkey)
        maxEpoch = max(maxEpoch, numel(epochData.(mkey).train_loss));
    end
end

% --- Training Loss ---
subplot(1,2,1); hold on; title('Training Loss vs Epoch');
xlabel('Epoch'); ylabel('Loss');
hLines1 = []; legNames1 = {};
% Torch models as curves
for i = 1:numel(torchKeys)
    mkey = char(torchKeys(i));
    idx = numel(sklearnKeys) + i;  % color index
    if isfield(epochData, mkey)
        h = plot(epochData.(mkey).train_loss, '-', 'LineWidth', 1.5, 'Color', cmap(idx,:));
        hLines1(end+1) = h;
        legNames1{end+1} = strrep(mkey, '_', ' ');
    end
end
% Sklearn models as horizontal dashed lines
for i = 1:numel(sklearnKeys)
    mkey = char(sklearnKeys(i));
    if isfield(sklearnMetrics, mkey)
        h = yline(sklearnMetrics.(mkey).train_loss, '--', 'Color', cmap(i,:), 'LineWidth', 1.5);
        hLines1(end+1) = h;
        legNames1{end+1} = strrep(mkey, '_', ' ');
    end
end
legend(hLines1, legNames1, 'Location', 'best', 'FontSize', 7); grid on; hold off;

% --- Validation Loss ---
subplot(1,2,2); hold on; title('Validation Loss vs Epoch');
xlabel('Epoch'); ylabel('Loss');
hLines2 = []; legNames2 = {};
for i = 1:numel(torchKeys)
    mkey = char(torchKeys(i));
    idx = numel(sklearnKeys) + i;
    if isfield(epochData, mkey)
        h = plot(epochData.(mkey).val_loss, '-', 'LineWidth', 1.5, 'Color', cmap(idx,:));
        hLines2(end+1) = h;
        legNames2{end+1} = strrep(mkey, '_', ' ');
    end
end
for i = 1:numel(sklearnKeys)
    mkey = char(sklearnKeys(i));
    if isfield(sklearnMetrics, mkey)
        h = yline(sklearnMetrics.(mkey).val_loss, '--', 'Color', cmap(i,:), 'LineWidth', 1.5);
        hLines2(end+1) = h;
        legNames2{end+1} = strrep(mkey, '_', ' ');
    end
end
legend(hLines2, legNames2, 'Location', 'best', 'FontSize', 7); grid on; hold off;

saveas(fig1, fullfile(outDir, "loss_vs_epoch.png"));
saveas(fig1, fullfile(outDir, "loss_vs_epoch.fig"));
fprintf("Saved loss_vs_epoch.png/.fig\n");

% ======================== Plot 2: Accuracy vs Epoch (all 8 models) ========================
fig2 = figure('Name', 'Accuracy vs Epoch', 'Position', [100 100 900 400]);

% --- Training Accuracy ---
subplot(1,2,1); hold on; title('Training Accuracy vs Epoch');
xlabel('Epoch'); ylabel('Accuracy');
hLines3 = []; legNames3 = {};
for i = 1:numel(torchKeys)
    mkey = char(torchKeys(i));
    idx = numel(sklearnKeys) + i;
    if isfield(epochData, mkey) && isfield(epochData.(mkey), 'train_acc')
        h = plot(epochData.(mkey).train_acc, '-', 'LineWidth', 1.5, 'Color', cmap(idx,:));
        hLines3(end+1) = h;
        legNames3{end+1} = strrep(mkey, '_', ' ');
    end
end
for i = 1:numel(sklearnKeys)
    mkey = char(sklearnKeys(i));
    if isfield(sklearnMetrics, mkey)
        h = yline(sklearnMetrics.(mkey).train_acc, '--', 'Color', cmap(i,:), 'LineWidth', 1.5);
        hLines3(end+1) = h;
        legNames3{end+1} = strrep(mkey, '_', ' ');
    end
end
legend(hLines3, legNames3, 'Location', 'best', 'FontSize', 7); grid on; hold off;

% --- Validation Accuracy ---
subplot(1,2,2); hold on; title('Validation Accuracy vs Epoch');
xlabel('Epoch'); ylabel('Accuracy');
hLines4 = []; legNames4 = {};
for i = 1:numel(torchKeys)
    mkey = char(torchKeys(i));
    idx = numel(sklearnKeys) + i;
    if isfield(epochData, mkey)
        h = plot(epochData.(mkey).val_acc, '-', 'LineWidth', 1.5, 'Color', cmap(idx,:));
        hLines4(end+1) = h;
        legNames4{end+1} = strrep(mkey, '_', ' ');
    end
end
for i = 1:numel(sklearnKeys)
    mkey = char(sklearnKeys(i));
    if isfield(sklearnMetrics, mkey)
        h = yline(sklearnMetrics.(mkey).val_acc, '--', 'Color', cmap(i,:), 'LineWidth', 1.5);
        hLines4(end+1) = h;
        legNames4{end+1} = strrep(mkey, '_', ' ');
    end
end
legend(hLines4, legNames4, 'Location', 'best', 'FontSize', 7); grid on; hold off;

saveas(fig2, fullfile(outDir, "accuracy_vs_epoch.png"));
saveas(fig2, fullfile(outDir, "accuracy_vs_epoch.fig"));
fprintf("Saved accuracy_vs_epoch.png/.fig\n");

% ======================== Plot 3: F1 vs Epoch (all 8 models) ========================
fig3 = figure('Name', 'F1 vs Epoch', 'Position', [100 100 900 400]);

% --- Training F1 ---
subplot(1,2,1); hold on; title('Training F1 vs Epoch');
xlabel('Epoch'); ylabel('F1');
hLines5 = []; legNames5 = {};
for i = 1:numel(torchKeys)
    mkey = char(torchKeys(i));
    idx = numel(sklearnKeys) + i;
    if isfield(epochData, mkey) && isfield(epochData.(mkey), 'train_f1')
        h = plot(epochData.(mkey).train_f1, '-', 'LineWidth', 1.5, 'Color', cmap(idx,:));
        hLines5(end+1) = h;
        legNames5{end+1} = strrep(mkey, '_', ' ');
    end
end
for i = 1:numel(sklearnKeys)
    mkey = char(sklearnKeys(i));
    if isfield(sklearnMetrics, mkey)
        h = yline(sklearnMetrics.(mkey).train_f1, '--', 'Color', cmap(i,:), 'LineWidth', 1.5);
        hLines5(end+1) = h;
        legNames5{end+1} = strrep(mkey, '_', ' ');
    end
end
legend(hLines5, legNames5, 'Location', 'best', 'FontSize', 7); grid on; hold off;

% --- Validation F1 ---
subplot(1,2,2); hold on; title('Validation F1 vs Epoch');
xlabel('Epoch'); ylabel('F1');
hLines6 = []; legNames6 = {};
for i = 1:numel(torchKeys)
    mkey = char(torchKeys(i));
    idx = numel(sklearnKeys) + i;
    if isfield(epochData, mkey)
        h = plot(epochData.(mkey).val_f1, '-', 'LineWidth', 1.5, 'Color', cmap(idx,:));
        hLines6(end+1) = h;
        legNames6{end+1} = strrep(mkey, '_', ' ');
    end
end
for i = 1:numel(sklearnKeys)
    mkey = char(sklearnKeys(i));
    if isfield(sklearnMetrics, mkey)
        h = yline(sklearnMetrics.(mkey).val_f1, '--', 'Color', cmap(i,:), 'LineWidth', 1.5);
        hLines6(end+1) = h;
        legNames6{end+1} = strrep(mkey, '_', ' ');
    end
end
legend(hLines6, legNames6, 'Location', 'best', 'FontSize', 7); grid on; hold off;

saveas(fig3, fullfile(outDir, "f1_vs_epoch.png"));
saveas(fig3, fullfile(outDir, "f1_vs_epoch.fig"));
fprintf("Saved f1_vs_epoch.png/.fig\n");

% ======================== Plot 4: Confusion Matrices ========================
nModels = numel(modelNames);
nCols = 4;
nRows = ceil(nModels / nCols);
fig4 = figure('Name', 'Confusion Matrices', 'Position', [50 50 1200 300*nRows]);

for i = 1:nModels
    yp = yProbTest{i};
    yPred = double(yp >= 0.5);
    tp = sum(yTest == 1 & yPred == 1);
    tn = sum(yTest == 0 & yPred == 0);
    fp = sum(yTest == 0 & yPred == 1);
    fn = sum(yTest == 1 & yPred == 0);
    cm = [tn fp; fn tp];
    acc = (tp + tn) / numel(yTest);
    f1  = binary_f1(yTest, yPred);

    subplot(nRows, nCols, i);
    imagesc(cm); colormap(gca, 'parula'); colorbar;
    set(gca, 'XTick', [1 2], 'XTickLabel', {'Pred 0','Pred 1'}, ...
             'YTick', [1 2], 'YTickLabel', {'True 0','True 1'});
    % Annotate cells
    for r = 1:2
        for c = 1:2
            text(c, r, sprintf('%d', cm(r,c)), ...
                'HorizontalAlignment', 'center', 'FontSize', 14, ...
                'FontWeight', 'bold', 'Color', 'w');
        end
    end
    titleStr = strrep(modelNames{i}, '_', ' ');
    title(sprintf('%s\nacc=%.3f  f1=%.3f', titleStr, acc, f1));
end

saveas(fig4, fullfile(outDir, "confusion_matrices.png"));
saveas(fig4, fullfile(outDir, "confusion_matrices.fig"));
fprintf("Saved confusion_matrices.png/.fig\n");

% ======================== Summary table ========================
fprintf("\n%-30s %8s %8s %8s %8s\n", "Model", "TrainAcc", "TrainF1", "TestAcc", "TestF1");
fprintf("%s\n", repmat('-', 1, 64));
for i = 1:nModels
    ypTr = yProbTrain{i};
    ypTe = yProbTest{i};
    trAcc = mean(double(ypTr >= 0.5) == yTrain);
    trF1  = binary_f1(yTrain, double(ypTr >= 0.5));
    teAcc = mean(double(ypTe >= 0.5) == yTest);
    teF1  = binary_f1(yTest, double(ypTe >= 0.5));
    fprintf("%-30s %8.4f %8.4f %8.4f %8.4f\n", modelNames{i}, trAcc, trF1, teAcc, teF1);
end

% ======================== Top 3 Recommendation ========================
% Rank by test accuracy, break ties with test F1, then fewer FP+FN
scores = zeros(nModels, 4);  % [testAcc, testF1, totalErrors, idx]
for i = 1:nModels
    ypTe = yProbTest{i};
    yPred = double(ypTe >= 0.5);
    teAcc = mean(yPred == yTest);
    teF1  = binary_f1(yTest, yPred);
    fp = sum(yTest == 0 & yPred == 1);
    fn = sum(yTest == 1 & yPred == 0);
    scores(i,:) = [teAcc, teF1, -(fp+fn), i];
end
% Sort descending by acc, then f1, then fewest errors
[~, order] = sortrows(scores, [-1 -2 -3]);
fprintf("\n==================== TOP 3 RECOMMENDED MODELS ====================\n");
for rank = 1:min(3, nModels)
    i = order(rank);
    ypTe = yProbTest{i};
    yPred = double(ypTe >= 0.5);
    teAcc = mean(yPred == yTest);
    teF1  = binary_f1(yTest, yPred);
    fp = sum(yTest == 0 & yPred == 1);
    fn = sum(yTest == 1 & yPred == 0);
    fprintf("  #%d  %-28s  acc=%.4f  f1=%.4f  FP=%d  FN=%d\n", ...
            rank, modelNames{i}, teAcc, teF1, fp, fn);
end
fprintf("===================================================================\n");

fprintf("\nAll outputs saved to: %s\n", outDir);
end

% ======================== Helper functions ========================

function [sIn, sOut] = generate_measurement_data(N)
sIn = struct();
idx = randperm(N);
nFront = round(0.5 * N);
idFront = idx(1:nFront);
idBack = idx(nFront+1:end);
sOut = struct();
sOut.isFront = false(N,1);
sOut.isFront(idFront) = true;
sIn.LER_dB = zeros(N,1);
sIn.LER_dB(idFront) = -15 + 4 * randn(numel(idFront),1);
sIn.LER_dB(idBack)  = -35 + 4 * randn(numel(idBack),1);
sIn.numPeaks = zeros(N,1);
sIn.numPeaks(idFront) = 1 + (rand(numel(idFront),1) > 0.8);
sIn.numPeaks(idBack)  = 2 - (rand(numel(idBack),1) > 0.8);
sIn.DFErr_deg = zeros(N,1);
sIn.DFErr_deg(idFront) = 2.0 * randn(numel(idFront),1);
sIn.DFErr_deg(idBack)  = 5.0 * randn(numel(idBack),1);
sIn.ENV_dB = zeros(N,1);
sIn.ENV_dB(idFront) = -12 + 5 * randn(numel(idFront),1);
sIn.ENV_dB(idBack)  = -8 + 7 * randn(numel(idBack),1);
sIn.SNR_dB = zeros(N,1);
sIn.SNR_dB(idFront) = 16 + 4 * randn(numel(idFront),1);
sIn.SNR_dB(idBack)  = 8 + 5 * randn(numel(idBack),1);
sIn.Freq_MHz = 95 + 10 * randn(N,1);
end

function X = measurements_to_X(sIn)
X = [double(sIn.LER_dB(:)), double(sIn.numPeaks(:)), ...
     double(sIn.DFErr_deg(:)), double(sIn.ENV_dB(:)), ...
     double(sIn.SNR_dB(:)), double(sIn.Freq_MHz(:))];
end

function f1 = binary_f1(yTrue, yPred)
yTrue = logical(yTrue(:));
yPred = logical(yPred(:));
tp = sum(yTrue & yPred);
fp = sum(~yTrue & yPred);
fn = sum(yTrue & ~yPred);
den = 2*tp + fp + fn;
if den == 0, f1 = 0.0; else, f1 = 2*tp / den; end
end

function L = bce_loss(y, p)
% Binary cross-entropy loss
ep = 1e-7;
p = max(min(p, 1-ep), ep);
L = -mean(y .* log(p) + (1 - y) .* log(1 - p));
end

function x = py_to_double_vector(pyObj)
arr = py.numpy.asarray(pyObj);
x = double(arr).';
x = x(:);
end
