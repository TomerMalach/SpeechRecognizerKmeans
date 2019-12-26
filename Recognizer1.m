clc;
clear all;

load data_training-test.mat

Fs = 16384; % 16KHz

%sound(training_data{1,1}, 16384);

MFCC = 0;
Training = 1;
TrainAccuracy = 1;
ValidationAccuracy = 1;
TestAccuracy = 1;

p = 24; % Num of LPC coeffs
Overlap = 0.5;
MFCC_coeffs_num = 13;
WindowsLength = 20*10^-3; % 30 msec


%% Cross varidation (train: 70%, val: 30%)
%cv = cvpartition(size(training_data, 2),'HoldOut',0.3);
%idx = cv.test;

% Separate to training and test data
dataTrain = training_data(:, 1:70);
dataVal  = training_data(:, 71:end);

NumberOfSamplesAtEachWindow = round(Fs * WindowsLength); 
StepSizeBetweenFrames = round(Overlap * NumberOfSamplesAtEachWindow);


%% Training

if Training

    Numbers = size(dataTrain, 1);
    Speakers = size(dataTrain, 2);

    NumsCodeBook = cell(Numbers, 1);
    SignalVecs = cell(Numbers, 1);
    
    %Centers = [256, 256, 512, 512, 512, 512, 512, 512, 512, 512]; % MFCC
    %Centers = [256, 512, 512, 512, 512, 1024, 1024, 1024, 512, 512]; % LPC
    %Centers = [256, 256, 256, 256, 256, 256, 256, 256, 256, 256]; % LPC
    
    Centers = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128];

    %Centers = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024];

    for num = 1:Numbers
    
        % Number of MFCC vecs per num
        NumOfVecs = 0;
        for speaker = 1:Speakers
            [StartPoint, EndPoint] = end_point_detect(dataTrain{num,speaker}, Fs, 0);
            SignalLength = length(dataTrain{num,speaker}(StartPoint:EndPoint)) - NumberOfSamplesAtEachWindow + StepSizeBetweenFrames;
%            SignalLength = length(dataTrain{num,speaker}) - NumberOfSamplesAtEachWindow + StepSizeBetweenFrames;
            FramesNumberPerRec = fix((SignalLength)/StepSizeBetweenFrames);
            NumOfVecs = NumOfVecs + FramesNumberPerRec;
        end

        if MFCC
            SignalVecs{num} = zeros(MFCC_coeffs_num + 1, NumOfVecs);
        else
            SignalVecs{num} = zeros(p + 1, NumOfVecs); 
        end
        
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
                coeffs = squeeze(mfcc(FramesSig ,Fs, 'WindowLength', round(Fs*WindowsLength), 'OverlapLength', round(Fs*WindowsLength*0.8)));
            else
                % Get LPC coeffs & Cov vec (since the cov mat is toplitz
                % mat)
                coeffs = AutoCorrelationPerColumn(FramesSig, p);
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
    evaluate_model(dataTrain, NumsCodeBook, MFCC, 'Train', Fs, p, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);
end


%% Validation Accuracy

if ValidationAccuracy
    evaluate_model(dataVal, NumsCodeBook, MFCC, 'Validation', Fs, p, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);
end


%% Test Accuracy

if TestAccuracy
    evaluate_model(test_data, NumsCodeBook, MFCC, 'Test', Fs, p, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);
end






