% 35個隱藏神經元
% 600筆資料，前400筆訓練，後200筆測試
% 100個epoch

%--------------------------------------------------------------------------
outnet = [];

% 設定存取資料的陣列
out = [];
s = [];
y = [];
x1 = [];
x2 = [];
x3 = [];
x4 = [];

input = [];
target = [];
% 以隨機變數作為資料
for i = 1:1:600
    x1 = rand;
    x2 = rand;
    x3 = rand;
    x4 = rand;
    s = [x1, x2, x3, x4];
    input = [input;s];
    
    % 設定模擬的函數
    y = 0.8*x1*x2*x3*x4+x1.^2+x2.^2+x3.^3+x4.^2+x1+x2*0.7-x2.^2*x3.^2+0.5*x1*x4.^2+x4*x2.^3+(-x1)*x2+(x1*x2*x3*x4).^3+(x1-x2+x3-x4)+(x1*x4)-(x2*x3)-2;
    
    % 設定目標
    target = [target;y];

end

% initialize the weight matrix
outputmatrix = zeros(35, 1);
for i = 1:1:35
    for j = 1:1:1
        outputmatrix(i, j) = rand;
    end
end
hiddenmatrix = zeros(4, 35);
for i = 1:1:4
    for j = 1:1:35
        hiddenmatrix(i, j) = rand;
    end
end

%--------------------------------------------------------------------------
% 進行倒傳遞類經網路的計算
% 4個input，35個hidden neuron，1個output
% 1個input含600個值，對應target的600個值
% hiddenmatrix為1個input與35個hidden neuron的weight值
% outputmatrix為35個hidden neuron與output的weight值
inputNum = 4;
hiddenNum = 35;
trainNum = 400;
totalNum = 600;
trainInput = input(1:1:trainNum, :);
trainTarget = target(1:1:trainNum, :);
testInput = input(trainNum+1:1:totalNum, :);
testTarget = target(trainNum+1:1:totalNum, :);
RMSE = [];
testRMSE = [];

for epoch = 1:1:100

for t = 1:1:trainNum
    
% 訓練
%--------------------------------------------------------------------------
% 1 x 35 = 1 x 4 * 4 x 35
Tmp1 = logsig(trainInput(t, :) * hiddenmatrix);
% 1 x 1 = 1 x 35 * 35 x 1
Tmp2 = purelin(Tmp1 * outputmatrix);
% 1 x 1 = 1 x 1 * 1 x 1
delta = (trainTarget(t, :) - Tmp2) * dpurelin(Tmp2);
% 1 x 35
delta =  delta * Tmp1;
% 35 x 1 = 35 x 1 + 1 x 35
newoutput = outputmatrix + delta.' * 0.1;
% 35 x 1 = 1 x 1 * 35 x 1
delta =  (trainTarget(t, :) - Tmp2) * outputmatrix ;
% 1 x 35
temp = dlogsig(Tmp1,  Tmp1) .* logsig(Tmp1);
% 1 x 1 =  4 x 1 * 1 x 35 * 35 x 1* 1 x 35 
delta =  trainInput(t, :).' * temp *  delta * Tmp1;
% 4 x 35 = 4 x 35 + 4 x 35
newhidden = hiddenmatrix + delta * 0.01;

% 將更新值指定至原先變數
outputmatrix = newoutput;
hiddenmatrix = newhidden;

end

% 測試
%--------------------------------------------------------------------------

Train1 = logsig(trainInput * hiddenmatrix);
Train2 = purelin(Train1 * outputmatrix);
RMSE = [RMSE;sqrt(mean((trainTarget - Train2) .^ 2))];
testTmp1 = logsig(testInput * hiddenmatrix);
testTmp2 = purelin(testTmp1 * outputmatrix);
testRMSE = [testRMSE;sqrt(mean((testTarget - testTmp2) .^ 2))];

end

%--------------------------------------------------------------------------
% 繪出結果(Figure 1, Figure 2)
% Figure 2 為第401~600筆資料的顯示結果，為200個點，將所有點連線的結果
% 繪製Figure 1
figure(1);
x = [];
for k = 1:1:100
    x = [x;k];
end
xx = linspace(0, 100);
y = RMSE(:, 1);
plot(x, y);
hold on;
xlabel("Epoch");
ylabel("RMSE");
y = testRMSE;
plot(x, y);
legend('Training', 'Simulation');

% 繪製Figure 2

figure(2);
x = [];
for k = 401:1:600
    x = [x;k];
end
y = testTarget;
plot(x, y);
hold on;
y = testTmp2;
plot(x, y);
legend('Function', 'Simulation');

