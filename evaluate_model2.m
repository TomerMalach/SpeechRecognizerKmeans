function [Accuracy] = evaluate_model2(data, CB, type, Fs, NumCoeffs, WindowsLength, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames)

    Numbers = size(data, 1);
    Speakers = size(data, 2);

    Prediction = zeros(Speakers, 1);
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

            % Get MFCC coeffs
            coeffs = squeeze(mfcc(FramesSig ,Fs, 'WindowLength', round(Fs*WindowsLength), 'OverlapLength', round(Fs*WindowsLength*0.8), 'NumCoeffs', NumCoeffs));
            
            cb_dist = [];
            for i=1:Numbers
                cb_dist = [cb_dist min(dist(coeffs, CB{i}, 1), [], 2)];
            end

            [~, argmin] = min(cb_dist, [], 2);
            Prediction(speaker) = mode(argmin);
        end

        Accuracy(num) = sum(Prediction == num)/length(data(num, :));
        
        display([type ' Accuracy for number ' num2str(num-1) ' is ' num2str(Accuracy(num))] );

    end

    disp([type ' Accuracy:']);
    disp(mean(Accuracy));

end

