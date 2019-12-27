clc;
clear all;

load data_training-test.mat

Fs = 16384; % 16KHz

%sound(training_data{1,1}, 16384);

OneCB = 1;
MFCC = 1;
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

NumberOfSamplesAtEachWindow = round(Fs * WindowsLength); # 30ms worth of samples according to the sample freq
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
    
    Centers = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128]; % num of centers at each codebook

    %Centers = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024];

    for num = 1:Numbers  
        % Evaluate the number of MFCC/LPC vecs per number
        NumOfVecs = 0;
        for speaker = 1:Speakers
            % clear the "empty" parts at the start/end of the signal (with margin to make sure we dont miss out on important data)
            [StartPoint, EndPoint] = end_point_detect(dataTrain{num,speaker}, Fs, 0);
            % (?) the signal length is calculated in this way to get a correct assessment of how much frames the signal is going to be divided to (?)
            SignalLength = length(dataTrain{num,speaker}(StartPoint:EndPoint)) - NumberOfSamplesAtEachWindow + StepSizeBetweenFrames;
            % SignalLength = length(dataTrain{num,speaker}) - NumberOfSamplesAtEachWindow + StepSizeBetweenFrames;
            FramesNumberPerRec = fix( (SignalLength) / StepSizeBetweenFrames);
            NumOfVecs = NumOfVecs + FramesNumberPerRec;
        end
        % data structure to hold the MFCC\LPC representation for each recording
        if MFCC
            SignalVecs{num} = zeros(MFCC_coeffs_num + 1, NumOfVecs);
        else
            SignalVecs{num} = zeros(p + 1, NumOfVecs); 
        end
        
        VecOffset = 1;

        for speaker = 1:Speakers
            
            % Edge Detector - throwing away "empty" parts of the signal at start\end with margin
            [StartPoint, EndPoint] = end_point_detect(dataTrain{num,speaker}, Fs, 0);
            dataTrain{num,speaker} = dataTrain{num,speaker}(StartPoint:EndPoint);
            
            % Framing - segmenting the signal into overlapping frames
            FramesSig = enframe(dataTrain{num,speaker}, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);

            % Hamming Window - multiplying each frame by Hamming Window, because we want to minimize the artificial sidelobes and get a cleaner result in the spectral domain
            NumberOfFrame = size(FramesSig, 1);
            HammingWindow = hamming(NumberOfSamplesAtEachWindow); % how much windows to create
            FramesSig = (FramesSig .* repmat(HammingWindow', NumberOfFrame, 1))';           
            
            if MFCC
                % Get MFCC coeffs, Note that WindowLength or OverlapLength arg is in samples
                coeffs = squeeze(mfcc(FramesSig ,Fs, 'WindowLength', round(Fs*WindowsLength), 'OverlapLength', round(Fs*WindowsLength*0.8)));
            else
                % Get LPC(?) coeffs & Cov vec (since the cov mat is toplitz
                % mat)
                coeffs = AutoCorrelationPerColumn(FramesSig, p); % (?)
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
        evaluate_model2(dataTrain, NumsCodeBook, 'Train', Fs, WindowsLength, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);
    else
        evaluate_model(dataTrain, NumsCodeBook, MFCC, 'Train', Fs, p, WindowsLength, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);
    end
end


%% Validation Accuracy

if ValidationAccuracy
    if OneCB
        evaluate_model2(dataVal, NumsCodeBook, 'Validation', Fs, WindowsLength, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);
    else
        evaluate_model(dataVal, NumsCodeBook, MFCC, 'Validation', Fs, p, WindowsLength, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);
    end
end


%% Test Accuracy

if TestAccuracy
    if OneCB
        evaluate_model2(test_data, NumsCodeBook, 'Test', Fs, WindowsLength, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);
    else
        evaluate_model(test_data, NumsCodeBook, MFCC, 'Test', Fs, p, WindowsLength, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);
    end
end






