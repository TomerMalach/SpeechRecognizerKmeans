function [] = evaluate_model(data, CB, MFCC, type, Fs, p, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames)

    Numbers = size(data, 1);
    Speakers = size(data, 2);

    Prediction = zeros(Numbers, Speakers);
    Prediction(1, :) = inf; 
    Accuracy = zeros(Numbers, 1);

    for num = 1:Numbers

        for speaker = 1:Speakers

            % Edge Detector
            [StartPoint, EndPoint] = end_point_detect(data{num,speaker}, Fs, 0);

            % Framing
            FramesSig = enframe(data{num,speaker}(StartPoint:EndPoint), NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);

            % Hamming Window
            NumberOfFrame = size(FramesSig, 1);
            HammingWindow = hamming(NumberOfSamplesAtEachWindow); % how much windows to create
            FramesSig = (FramesSig .* repmat(HammingWindow', NumberOfFrame, 1))'; 

            if MFCC
                % Get MFCC coeffs
                coeffs = squeeze(mfcc(FramesSig ,Fs, 'WindowLength', round(Fs*WindowsLength), 'OverlapLength', round(Fs*WindowsLength*0.8)));
            else
                % Get AutoCorrelation
                coeffs = AutoCorrelationPerColumn(FramesSig, p);
            end
            
            for i=1:Numbers
                Prediction(i, speaker) = sum(min(dist(coeffs, CB{i}, MFCC), [], 2));
            end
        end

        [~, argmin] = min(Prediction, [], 1);
        Accuracy(num) = sum(argmin == num)/length(data(num, :));
        
        display([type ' Accuracy for number ' num2str(num-1) ' is ' num2str(Accuracy(num))] );

    end

    disp([type ' Accuracy:']);
    disp(mean(Accuracy));

end

