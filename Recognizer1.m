clc;
clear all;

load data_training-test.mat

Fs = 16384; % 16KHz

%sound(training_data{1,1}, 16384);

OneCB = 0;
MFCC = 0;
Training = 1;
TrainAccuracy = 1;
ValidationAccuracy = 1;
TestAccuracy = 1;

p = 24; % Num of LPC coeffs
Overlap = 0.5;
MFCCs = 13;
WindowsLength = 10*10^-3; % msec

BestParamModelAccuracy = 0;
BestParamModel = [];

%% Separate to training and val data
% Cross varidation (train: 70%, val: 30%)
%cv = cvpartition(size(training_data, 2),'HoldOut',0.3);
%idx = cv.test;

dataTrain = training_data(:, 1:70);
dataVal  = training_data(:, 71:end);

NumberOfSamplesAtEachWindow = round(Fs * WindowsLength); 
StepSizeBetweenFrames = round(Overlap * NumberOfSamplesAtEachWindow);

if MFCC
    NumCoeffs = MFCCs;
else
    NumCoeffs = p;
end


%% Training

if Training

    Numbers = size(dataTrain, 1);
    Speakers = size(dataTrain, 2);

    NumsCodeBook = cell(Numbers, 1);
    SignalVecs = cell(Numbers, 1);
        
    Centers = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128];

    for num = 1:Numbers
    
        % Number of MFCC vecs per num
        NumOfVecs = 0;
        for speaker = 1:Speakers
            [StartPoint, EndPoint] = end_point_detect(dataTrain{num,speaker}, Fs, 0);
            SignalLength = length(dataTrain{num,speaker}(StartPoint:EndPoint)) - NumberOfSamplesAtEachWindow + StepSizeBetweenFrames;
            FramesNumberPerRec = fix((SignalLength)/StepSizeBetweenFrames);
            NumOfVecs = NumOfVecs + FramesNumberPerRec;
        end

        SignalVecs{num} = zeros(NumCoeffs + 1, NumOfVecs);
        
        VecOffset = 1;

        for speaker = 1:Speakers

            % Edge Detector
            [StartPoint, EndPoint] = end_point_detect(dataTrain{num,speaker}, Fs, 0);
            dataTrain{num,speaker} = dataTrain{num,speaker}(StartPoint:EndPoint);
            
            % Framing
            FramesSig = enframe(dataTrain{num,speaker}, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);

            % Hamming Window
            NumberOfFrame = size(FramesSig, 1);
            HammingWindow = hamming(NumberOfSamplesAtEachWindow); % how much windows to create
            FramesSig = (FramesSig .* repmat(HammingWindow', NumberOfFrame, 1))';           
            
            if MFCC
                % Get MFCC coeffs
                coeffs = squeeze(mfcc(FramesSig ,Fs, 'WindowLength', round(Fs*WindowsLength), 'OverlapLength', round(Fs*WindowsLength*0.8), 'NumCoeffs', NumCoeffs));
            else
                % Get cov vec (the cov mat it is toplitz mat)
                coeffs = AutoCorrelationPerColumn(FramesSig, NumCoeffs);
            end
            
            SignalVecs{num}(:, VecOffset:(VecOffset + size(coeffs, 2) - 1)) = coeffs;              
            VecOffset = VecOffset + size(coeffs,2);
        end

        NumsCodeBook{num} = vqlbg(SignalVecs{num}, Centers(num), MFCC);
        
        display(['CodeBook for number ' num2str(num-1) ' is ready!']);

    end   
    
    save(['CB_' num2str(MFCC) '_' num2str(mean(Centers)) '_' datestr(now,'dd-mm-yy_HH-MM') '.mat'], 'NumsCodeBook');

end


%% Train Accuracy
if TrainAccuracy
    if OneCB
        TrainAccuracyVals = evaluate_model2(dataTrain, NumsCodeBook, 'Train', Fs, NumCoeffs, WindowsLength, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);
    else
        TrainAccuracyVals = evaluate_model(dataTrain, NumsCodeBook, MFCC, 'Train', Fs, NumCoeffs, WindowsLength, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);
    end
end


%% Validation Accuracy

if ValidationAccuracy
    if OneCB
        ValAccuracyVals = evaluate_model2(dataVal, NumsCodeBook, 'Validation', Fs, NumCoeffs, WindowsLength, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);
    else
        ValAccuracyVals = evaluate_model(dataVal, NumsCodeBook, MFCC, 'Validation', Fs, NumCoeffs, WindowsLength, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);
    end
end


%% Test Accuracy

if TestAccuracy
    if OneCB
        TestAccuracyVals = evaluate_model2(test_data, NumsCodeBook, 'Test', Fs, NumCoeffs, WindowsLength, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);
    else
        TestAccuracyVals = evaluate_model(test_data, NumsCodeBook, MFCC, 'Test', Fs, NumCoeffs, WindowsLength, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);
    end
end

%% Write Accuracy To File

fileID = fopen(['EXP_ONECB' num2str(OneCB) '_MFCC' num2str(MFCC) 
    '_NumCoeffs' num2str(NumCoeffs) '_Overlap' num2str(Overlap*100) '.txt'],'w');
if TrainAccuracy
    fprintf(fileID,'Training:f\r\n');
    fprintf(fileID,'%12.8f\r\n', TrainAccuracyVals);
    fprintf(fileID,'%Total: 12.8f\r\n', mean(TrainAccuracyVals));
end
if ValAccuracy
    fprintf(fileID,'Validation:f\r\n');
    fprintf(fileID,'%12.8f\r\n', ValAccuracyVals);
    fprintf(fileID,'%Total: 12.8f\r\n', mean(ValAccuracyVals));
end
if TestAccuracy
    fprintf(fileID,'Test:f\r\n');
    fprintf(fileID,'%12.8f\r\n', TestAccuracyVals);
    fprintf(fileID,'%Total: 12.8f\r\n', mean(TestAccuracyVals));
end
fclose(fileID);

if mean(ValAccuracyVals) > BestParamModelAccuracy
    BestParamModelAccuracy = mean(ValAccuracyVals);
    BestParamModel = ['EXP_ONECB' num2str(OneCB) '_MFCC' num2str(MFCC) 
    '_NumCoeffs' num2str(NumCoeffs) '_Overlap' num2str(Overlap*100) '.txt'];
end

