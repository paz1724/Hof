function test_all_models_matlab()
% Test every model mode through Train_and_predict and verify that
% Python print() output appears in MATLAB's command window.
%
% Run:  matlab -batch "cd('C:\GitHub\Hof'); test_all_models_matlab"

rng(42);
N = 120;  % enough samples for stratified splits
F = 5;

% --- Generate synthetic balanced data ---
X = randn(N, F);
y = [zeros(N/2,1); ones(N/2,1)];
idx = randperm(N);
X = X(idx,:);
y = y(idx);

pythonExe = "";  % use default Python
seed = 0;

% =====================================================================
% 1. Default models (no models field) → random_forest, extra_trees, hist_gradient_boosting
% =====================================================================
fprintf("\n===== TEST 1: Default models (random_forest, extra_trees, hist_gradient_boosting) =====\n");
sIn = struct('X', X, 'y', y);
res = Train_and_predict(sIn, pythonExe, seed);
chosen = string(res{"chosen"});
fprintf("  chosen = %s\n", chosen);
assert(chosen ~= "", "chosen must be non-empty");
assert(logical(py.operator.contains(res, "tree")), "Must have tree result");
fprintf("  PASS\n");

% =====================================================================
% 2. SVM only
% =====================================================================
fprintf("\n===== TEST 2: SVM only =====\n");
sIn2 = struct('X', X, 'y', y, 'models', ["svm"]);
res2 = Train_and_predict(sIn2, pythonExe, seed);
chosen2 = string(res2{"chosen"});
fprintf("  chosen = %s\n", chosen2);
assert(logical(py.operator.contains(res2, "tree")), "Must have tree result for SVM");
model_name = string(res2{"tree"}{"model"});
fprintf("  tree.model = %s\n", model_name);
assert(model_name == "svm", "Model name must be svm");
fprintf("  PASS\n");

% =====================================================================
% 3. CNN only
% =====================================================================
fprintf("\n===== TEST 3: CNN only =====\n");
sIn3 = struct('X', X, 'y', y, 'models', ["cnn"]);
res3 = Train_and_predict(sIn3, pythonExe, seed);
chosen3 = string(res3{"chosen"});
fprintf("  chosen = %s\n", chosen3);
assert(logical(py.operator.contains(res3, "cnn")), "Must have cnn result");
fprintf("  PASS\n");

% =====================================================================
% 4. Transformer only
% =====================================================================
fprintf("\n===== TEST 4: Transformer only =====\n");
sIn4 = struct('X', X, 'y', y, 'models', ["transformer"]);
res4 = Train_and_predict(sIn4, pythonExe, seed);
chosen4 = string(res4{"chosen"});
fprintf("  chosen = %s\n", chosen4);
assert(logical(py.operator.contains(res4, "transformer")), "Must have transformer result");
fprintf("  PASS\n");

% =====================================================================
% 5. RL only
% =====================================================================
fprintf("\n===== TEST 5: RL (DQN) only =====\n");
sIn5 = struct('X', X, 'y', y, 'models', ["rl"]);
res5 = Train_and_predict(sIn5, pythonExe, seed);
chosen5 = string(res5{"chosen"});
fprintf("  chosen = %s\n", chosen5);
assert(logical(py.operator.contains(res5, "rl")), "Must have rl result");
fprintf("  PASS\n");

% =====================================================================
% 6. Bagged Trees only
% =====================================================================
fprintf("\n===== TEST 6: Bagged Trees only =====\n");
sIn6 = struct('X', X, 'y', y, 'models', ["bagged_trees"]);
res6 = Train_and_predict(sIn6, pythonExe, seed);
chosen6 = string(res6{"chosen"});
fprintf("  chosen = %s\n", chosen6);
assert(logical(py.operator.contains(res6, "tree")), "Must have tree result for bagged_trees");
model_name6 = string(res6{"tree"}{"model"});
fprintf("  tree.model = %s\n", model_name6);
assert(model_name6 == "bagged_trees", "Model name must be bagged_trees");
fprintf("  PASS\n");

% =====================================================================
% 7. Mixed: random_forest + svm + cnn
% =====================================================================
fprintf("\n===== TEST 7: Mixed (random_forest + svm + cnn) =====\n");
sIn7 = struct('X', X, 'y', y, 'models', ["random_forest", "svm", "cnn"]);
res7 = Train_and_predict(sIn7, pythonExe, seed);
chosen7 = string(res7{"chosen"});
fprintf("  chosen = %s\n", chosen7);
assert(chosen7 ~= "", "chosen must be non-empty");
fprintf("  PASS\n");

% =====================================================================
% 8. All tabular models together
% =====================================================================
fprintf("\n===== TEST 8: All tabular models =====\n");
sIn8 = struct('X', X, 'y', y, 'models', ["random_forest", "extra_trees", "hist_gradient_boosting", "bagged_trees", "svm", "cnn", "transformer", "rl"]);
res8 = Train_and_predict(sIn8, pythonExe, seed);
chosen8 = string(res8{"chosen"});
fprintf("  chosen = %s\n", chosen8);
y_prob = py_to_double_vector(res8{"y_prob"});
fprintf("  y_prob length = %d (expected %d)\n", numel(y_prob), N);
assert(numel(y_prob) == N, "y_prob length must match N");
acc = double(res8{"metrics"}{"acc"});
f1  = double(res8{"metrics"}{"f1"});
fprintf("  acc=%.3f  f1=%.3f\n", acc, f1);
fprintf("  PASS\n");

% =====================================================================
fprintf("\n===== ALL %d TESTS PASSED =====\n", 8);
end

function x = py_to_double_vector(pyObj)
arr = py.numpy.asarray(pyObj);
x = double(arr).';
x = x(:);
end
