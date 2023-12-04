classdef imageProcess < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                       matlab.ui.Figure
        PepperValueEditField           matlab.ui.control.NumericEditField
        PepperValueEditFieldLabel      matlab.ui.control.Label
        SaltValueEditField             matlab.ui.control.NumericEditField
        SaltValueEditFieldLabel        matlab.ui.control.Label
        ErlangRateEditField            matlab.ui.control.NumericEditField
        ErlangRateEditFieldLabel       matlab.ui.control.Label
        ErlangShapeEditField           matlab.ui.control.NumericEditField
        ErlangShapeEditFieldLabel      matlab.ui.control.Label
        ExponentialRateEditField       matlab.ui.control.NumericEditField
        ExponentialRateEditFieldLabel  matlab.ui.control.Label
        ScaleEditField                 matlab.ui.control.NumericEditField
        ScaleEditFieldLabel            matlab.ui.control.Label
        MeanEditField                  matlab.ui.control.NumericEditField
        MeanEditFieldLabel             matlab.ui.control.Label
        AmplitudeEditField             matlab.ui.control.NumericEditField
        AmplitudeEditFieldLabel        matlab.ui.control.Label
        ProcessButton                  matlab.ui.control.Button
        SaveImageButton                matlab.ui.control.Button
        Image2                         matlab.ui.control.Image
        Image                          matlab.ui.control.Image
        StandartDeviationEditField     matlab.ui.control.NumericEditField
        StandartDeviationEditFieldLabel  matlab.ui.control.Label
        MaxWindowSizeEditField         matlab.ui.control.NumericEditField
        MaxWindowSizeEditFieldLabel    matlab.ui.control.Label
        KernelSizeEditField            matlab.ui.control.NumericEditField
        KernelSizeEditFieldLabel       matlab.ui.control.Label
        EditedImageLabel               matlab.ui.control.Label
        SelectedImageLabel             matlab.ui.control.Label
        SelectOperationListBox         matlab.ui.control.ListBox
        SelectOperationListBoxLabel    matlab.ui.control.Label
        OpenImageButton                matlab.ui.control.Button
    end

    methods (Access = private)
        function performHistogramEqualization(app)
            % Get the input image
            inputImage = rgb2gray(app.Image.ImageSource);
        
            % Compute the histogram of the input image
            histogram = imhist(inputImage);
        
            % Compute the cumulative distribution function (CDF) of the histogram
            cdf = cumsum(histogram) / numel(inputImage);
        
            % Perform histogram equalization
            outputImage = uint8(255 * cdf(inputImage + 1));
        
            % Display the edited image in the UI using the Image2 component
            app.Image2.ImageSource = cat(3, outputImage, outputImage, outputImage);
                end

        function performMeanFiltering(app, kernelSize)
            % Get the input image
            inputImage = rgb2gray(app.Image.ImageSource);
        
            % Pad the image to handle border pixels
            paddedImage = padarray(inputImage, [floor(kernelSize/2), floor(kernelSize/2)], 'replicate');
        
            % Create the output image
            outputImage = zeros(size(inputImage));
        
            % Apply mean filtering
            for i = 1:size(inputImage, 1)
                for j = 1:size(inputImage, 2)
                    neighborhood = paddedImage(i:i+kernelSize-1, j:j+kernelSize-1);
                    outputImage(i, j) = mean(neighborhood, 'all');
                end
            end
        
            % Display the edited image in the UI using the Image2 component
            app.Image2.ImageSource = cat(3, uint8(outputImage), uint8(outputImage), uint8(outputImage));
        end

        function performMedianFiltering(app, kernelSize)
            % Get the input image
            inputImage = rgb2gray(app.Image.ImageSource);
        
            % Pad the image to handle border pixels
            paddedImage = padarray(inputImage, [floor(kernelSize/2), floor(kernelSize/2)], 'replicate');
        
            % Create the output image
            outputImage = zeros(size(inputImage));
        
            % Apply median filtering
            for i = 1:size(inputImage, 1)
                for j = 1:size(inputImage, 2)
                    neighborhood = paddedImage(i:i+kernelSize-1, j:j+kernelSize-1);
                    outputImage(i, j) = median(neighborhood, 'all');
                end
            end
        
            % Display the edited image in the UI using the Image2 component
            app.Image2.ImageSource = cat(3, uint8(outputImage), uint8(outputImage), uint8(outputImage));
        end

        function performAdaptiveMedianFiltering(app, maxWindowSize)
            % Get the input image
            inputImage = rgb2gray(app.Image.ImageSource);
        
            % Pad the image to handle border pixels
            paddedImage = padarray(inputImage, [maxWindowSize, maxWindowSize], 'replicate');
        
            % Create the output image
            outputImage = zeros(size(inputImage));
        
            % Apply adaptive median filtering
            for i = 1:size(inputImage, 1)
                for j = 1:size(inputImage, 2)
                    outputImage(i, j) = adaptiveMedianFilter(paddedImage, i, j, maxWindowSize);
                end
            end
        
            % Display the edited image in the UI using the Image2 component
            app.Image2.ImageSource = cat(3, uint8(outputImage), uint8(outputImage), uint8(outputImage));
        
            % Inner function for adaptive median filtering
            function outputPixel = adaptiveMedianFilter(inputImage, i, j, maxWindowSize)
                windowSize = 3; % Initial window size
        
                while windowSize <= maxWindowSize
                    neighborhood = inputImage(i:i+windowSize-1, j:j+windowSize-1);
        
                    Zmin = min(neighborhood(:));
                    Zmax = max(neighborhood(:));
                    Zxy = inputImage(i, j);
        
                    if Zmin < Zxy && Zxy < Zmax
                        outputPixel = Zxy;
                        return;
                    end
        
                    windowSize = windowSize + 2;
                end
        
                % If no suitable window is found, return the median of the entire neighborhood
                outputPixel = median(neighborhood(:));
            end
        end

        function performGaussianSmoothing(app, sigma, kernelSize)
            % Get the input image
            inputImage = rgb2gray(app.Image.ImageSource);
        
            % Create a Gaussian kernel
            x = linspace(-floor(kernelSize/2), floor(kernelSize/2), kernelSize);
            gaussianKernel1D = exp(-x.^2 / (2 * sigma^2)) / (sqrt(2 * pi) * sigma);
            
            % Create a 2D Gaussian kernel
            gaussianKernel = gaussianKernel1D' * gaussianKernel1D;
            gaussianKernel = gaussianKernel / sum(gaussianKernel(:)); % Normalize the kernel
        
            % Pad the image to handle border pixels
            paddedImage = padarray(inputImage, [floor(kernelSize/2), floor(kernelSize/2)], 'replicate');
        
            % Create the output image
            outputImage = zeros(size(inputImage));
        
            % Apply Gaussian smoothing
            for i = 1:size(inputImage, 1)
                for j = 1:size(inputImage, 2)
                    neighborhood = paddedImage(i:i+kernelSize-1, j:j+kernelSize-1);
            
                    % Convert neighborhood to double
                    neighborhood = double(neighborhood);
            
                    % Perform element-wise multiplication and sum
                    outputImage(i, j) = sum(neighborhood(:) .* gaussianKernel(:));
                end
            end
        
            % Display the edited image in the UI using the Image2 component
            app.Image2.ImageSource = cat(3, uint8(outputImage), uint8(outputImage), uint8(outputImage));
        end

        function performAdditiveUniformNoise(app, amplitude)
            % Get the input image
            inputImage = rgb2gray(app.Image.ImageSource);
        
            % Generate uniform noise with the same size as the input image
            noise = amplitude * (rand(size(inputImage)) - 0.5);
        
            % Add the generated noise to the input image
            outputImage = double(inputImage) + noise;
        
            % Clip values to the valid intensity range [0, 255]
            outputImage = min(max(outputImage, 0), 255);
        
            % Display the edited image in the UI using the Image2 component
            app.Image2.ImageSource = cat(3, uint8(outputImage), uint8(outputImage), uint8(outputImage));
        end

        function performAdditiveGaussianNoise(app, meanValue, standardDeviation)
            % Get the input image
            inputImage = rgb2gray(app.Image.ImageSource);
        
            % Generate Gaussian noise with the same size as the input image
            noise = meanValue + standardDeviation * randn(size(inputImage));
        
            % Add the generated noise to the input image
            outputImage = double(inputImage) + noise;
        
            % Clip values to the valid intensity range [0, 255]
            outputImage = min(max(outputImage, 0), 255);
        
            % Display the edited image in the UI using the Image2 component
            app.Image2.ImageSource = cat(3, uint8(outputImage), uint8(outputImage), uint8(outputImage));
        end

        function performAdditiveSaltAndPepperNoise(app, black, white)
            % Get the input image
            inputImage = rgb2gray(app.Image.ImageSource);
        
            % Make a copy of the input image
            outputImage = inputImage;
        
            % Assuming black pixel value is 4 and white pixel value is 251
            b = black;
            w = white;
        
            % Generate a random matrix of the same size as the image
            [m, n, ~] = size(inputImage);
            randomMatrix = randi([0, 255], m, n);
        
            % Add salt and pepper noise to the copy of the input image
            outputImage(randomMatrix <= b) = 0;      % Pepper noise
            outputImage(randomMatrix >= w) = 255;    % Salt noise
        
            % Display the edited image in the UI using the Image2 component
            app.Image2.ImageSource = cat(3, uint8(outputImage), uint8(outputImage), uint8(outputImage));
        end

        function performAdditiveLogNormalNoise(app, meanValue, standardDeviation)
            % Get the input image
            inputImage = rgb2gray(app.Image.ImageSource);
        
            % Generate log-normal noise with the same size as the input image
            noise = exp(meanValue + standardDeviation * randn(size(inputImage)));
        
            % Add the generated noise to the input image
            outputImage = double(inputImage) .* noise;
        
            % Clip values to the valid intensity range [0, 255]
            outputImage = min(max(outputImage, 0), 255);
        
            % Display the edited image in the UI using the Image2 component
            app.Image2.ImageSource = cat(3, uint8(outputImage), uint8(outputImage), uint8(outputImage));
        end

        function performAdditiveRayleighNoise(app, scale)
            % Get the input image
            inputImage = rgb2gray(app.Image.ImageSource);
        
            % Generate Rayleigh noise with the same size as the input image
            noise = scale * sqrt(-2 * log(1 - rand(size(inputImage))));
        
            % Add the generated noise to the input image
            outputImage = double(inputImage) + noise;
        
            % Clip values to the valid intensity range [0, 255]
            outputImage = min(max(outputImage, 0), 255);
        
            % Display the edited image in the UI using the Image2 component
            app.Image2.ImageSource = cat(3, uint8(outputImage), uint8(outputImage), uint8(outputImage));
        end

        function performAdditiveExponentialNoise(app, lambda)
            % Get the input image
            inputImage = rgb2gray(app.Image.ImageSource);
        
            % Generate exponential noise with the same size as the input image
            noise = -1 / lambda * log(1 - rand(size(inputImage)));
        
            % Add the generated noise to the input image
            outputImage = double(inputImage) + noise;
        
            % Clip values to the valid intensity range [0, 255]
            outputImage = min(max(outputImage, 0), 255);
        
            % Scale the values for display in the uint8 format
            outputImage = uint8((outputImage / 255) * 255);
        
            % Display the edited image in the UI using the Image2 component
            app.Image2.ImageSource = cat(3, outputImage, outputImage, outputImage);
        end

        function performAdditiveErlangNoise(app, k, lambda)
            % Get the input image
            inputImage = rgb2gray(app.Image.ImageSource);
        
            % Generate Erlang-distributed noise with the same size as the input image
            noise = -1 / lambda * log(1 - rand(size(inputImage)) * k);
        
            % Add the generated noise to the input image
            outputImage = double(inputImage) + noise;
        
            % Clip values to the valid intensity range [0, 255]
            outputImage = min(max(outputImage, 0), 255);
        
            % Display the edited image in the UI using the Image2 component
            app.Image2.ImageSource = cat(3, uint8(outputImage), uint8(outputImage), uint8(outputImage));
        end

    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Button pushed function: OpenImageButton
        function OpenImageButtonPushed(app, event)
            % Open a file dialog for selecting a PGM file
            [filename, pathname] = uigetfile({'*.pgm'}, 'Select a PGM file');

            % Check if the user selected a file
            if ~isequal(filename, 0)
                % Load the selected image
                img = imread(fullfile(pathname, filename));

                % Convert the grayscale image to an RGB format
                imgRGB = cat(3, img, img, img);
        
                % Display the image in the UI using the Image component
                app.Image.ImageSource = imgRGB;
            end
        end

        % Button pushed function: ProcessButton
        function ProcessButtonPushed(app, event)
            % Get the selected operation from the listbox
            selectedOperation = app.SelectOperationListBox.Value;
            % Handle the selected operation
            switch selectedOperation
                case 'Histogram Equalization'
                    app.performHistogramEqualization();

                case 'Mean (Average) Filtering'
                    % Call the performMeanFiltering function with the user-entered kernel size
                    kernelSize = app.KernelSizeEditField.Value;
                    app.performMeanFiltering(kernelSize);
        
                case 'Median Filtering'
                    % Call the performMedianFiltering function with the user-entered kernel size
                    kernelSize = app.KernelSizeEditField.Value;
                    app.performMedianFiltering(kernelSize);
        
                case 'Adaptive Median Filtering'
                    % Call the performAdaptiveMedianFiltering function with the user-entered maximum window size
                    maxWindowSize = app.MaxWindowSizeEditField.Value;
                    app.performAdaptiveMedianFiltering(maxWindowSize);

                case 'Gaussian Filtering'
                    % Get the standard deviation and kernel size from the UI 
                    sigma = app.StandartDeviationEditField.Value;
                    kernelSize = app.KernelSizeEditField.Value;
                
                    % Call the performGaussianSmoothing function with the given standard deviation and kernel size
                    app.performGaussianSmoothing(sigma, kernelSize);

                case 'Additive Uniform Noise'
                    % Get the amplitude from the UI 
                    amplitude = app.AmplitudeEditField.Value;
                
                    % Call the performAdditiveUniformNoise function with the given amplitude
                    app.performAdditiveUniformNoise(amplitude);

                case 'Additive Gaussian Noise'
                    % Get the mean and standard deviation from the UI 
                    meanValue = app.MeanEditField.Value;
                    standardDeviation = app.StandartDeviationEditField.Value;
                
                    % Call the performAdditiveGaussianNoise function with the given parameters
                    app.performAdditiveGaussianNoise(meanValue, standardDeviation);

                case 'Additive Salt&Pepper Noise'
                    % Get the density from the UI 
                    black = app.PepperValueEditField.Value;
                    white = app.SaltValueEditField.Value;
                
                    % Call the performAdditiveSaltAndPepperNoise function with the given density
                    app.performAdditiveSaltAndPepperNoise(black, white);

                case 'Additive LogNormal Noise'
                    % Get the mean and standard deviation from the UI
                    meanValue = app.MeanEditField.Value;
                    standardDeviation = app.StandartDeviationEditField.Value;
                
                    % Call the performAdditiveLogNormalNoise function with the given parameters
                    app.performAdditiveLogNormalNoise(meanValue, standardDeviation);

                case 'Additive Rayleigh Noise'
                    % Get the scale parameter from the UI 
                    scale = app.ScaleEditField.Value;
                    
                    % Call the performAdditiveRayleighNoise function with the given scale
                    app.performAdditiveRayleighNoise(scale); 
                
                case 'Additive Exponential Noise'
                    % Get the rate parameter from the UI 
                    lambda = app.ExponentialRateEditField.Value;
                    
                    % Call the performAdditiveExponentialNoise function with the given rate
                    app.performAdditiveExponentialNoise(lambda);

                case 'Additive Erlang Noise'
                    % Get the shape and rate parameters from the UI 
                    k = app.ErlangShapeEditField.Value;
                    lambda = app.ErlangRateEditField.Value;
                
                    % Call the performAdditiveErlangNoise function with the given parameters
                    app.performAdditiveErlangNoise(k, lambda);
            end 
        end

        % Button pushed function: SaveImageButton
        function SaveImageButtonPushed(app, event)
            % Get the edited image from the Image2 component
            editedImage = rgb2gray(app.Image2.ImageSource);
    
            % Open a save file dialog for selecting the save location
            [filename, pathname] = uiputfile({'*.pgm'}, 'Save Image As');
    
            % Check if the user selected a location
            if ~isequal(filename, 0)
                % Construct the full file path
                fullFilePath = fullfile(pathname, filename);
    
                % Write the edited image to the selected file location
                imwrite(editedImage, fullFilePath, 'pgm');
            end
        end

        % Value changed function: SelectOperationListBox
        function SelectOperationListBoxValueChanged(app, event)
            value = app.SelectOperationListBox.Value;
            switch value
                case 'Histogram Equalization'
                case 'Mean (Average) Filtering'
                   uialert(app.UIFigure, 'Please enter Kernel Size from down below.', 'Enter Parameters', 'Icon', 'warning');
                case 'Median Filtering'
                    uialert(app.UIFigure, 'Please enter Kernel Size from down below.', 'Enter Parameters', 'Icon', 'warning');
                case 'Adaptive Median Filtering'
                    uialert(app.UIFigure, 'Please enter Max Window Size from down below.', 'Enter Parameters', 'Icon', 'warning');
                case 'Gaussian Filtering'
                    uialert(app.UIFigure, 'Please enter Standart Deviation and Kernel Size from down below.', 'Enter Parameters', 'Icon', 'warning');
                case 'Additive Uniform Noise'
                    uialert(app.UIFigure, 'Please enter Amplitude from down below.', 'Enter Parameters', 'Icon', 'warning');
                case 'Additive Gaussian Noise'
                    uialert(app.UIFigure, 'Please enter Mean and Standart Deviation from down below.', 'Enter Parameters', 'Icon', 'warning');
                case 'Additive Salt&Pepper Noise'
                    uialert(app.UIFigure, 'Please enter Salt and Pepper Value from down below.', 'Enter Parameters', 'Icon', 'warning');
                case 'Additive LogNormal Noise'
                   uialert(app.UIFigure, 'Please enter Mean and Standart Deviation from down below.', 'Enter Parameters', 'Icon', 'warning');
                case 'Additive Rayleigh Noise'
                    uialert(app.UIFigure, 'Please enter Scale from down below.', 'Enter Parameters', 'Icon', 'warning');
                case 'Additive Exponential Noise'
                    uialert(app.UIFigure, 'Please enter Exponential Rate from down below.', 'Enter Parameters', 'Icon', 'warning');
                case 'Additive Erlang Noise'
                   uialert(app.UIFigure, 'Please enter Erlang Shape and Erlang Rate from down below.', 'Enter Parameters', 'Icon', 'warning');
            end 
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 854 717];
            app.UIFigure.Name = 'MATLAB App';

            % Create OpenImageButton
            app.OpenImageButton = uibutton(app.UIFigure, 'push');
            app.OpenImageButton.ButtonPushedFcn = createCallbackFcn(app, @OpenImageButtonPushed, true);
            app.OpenImageButton.Position = [27 668 100 23];
            app.OpenImageButton.Text = 'Open Image';

            % Create SelectOperationListBoxLabel
            app.SelectOperationListBoxLabel = uilabel(app.UIFigure);
            app.SelectOperationListBoxLabel.HorizontalAlignment = 'right';
            app.SelectOperationListBoxLabel.Position = [27 624 94 22];
            app.SelectOperationListBoxLabel.Text = 'Select Operation';

            % Create SelectOperationListBox
            app.SelectOperationListBox = uilistbox(app.UIFigure);
            app.SelectOperationListBox.Items = {'Histogram Equalization', 'Mean (Average) Filtering', 'Median Filtering', 'Adaptive Median Filtering', 'Gaussian Filtering', 'Additive Uniform Noise', 'Additive Gaussian Noise', 'Additive Salt&Pepper Noise', 'Additive LogNormal Noise', 'Additive Rayleigh Noise', 'Additive Exponential Noise', 'Additive Erlang Noise'};
            app.SelectOperationListBox.ValueChangedFcn = createCallbackFcn(app, @SelectOperationListBoxValueChanged, true);
            app.SelectOperationListBox.Position = [136 509 192 139];
            app.SelectOperationListBox.Value = 'Histogram Equalization';

            % Create SelectedImageLabel
            app.SelectedImageLabel = uilabel(app.UIFigure);
            app.SelectedImageLabel.Position = [427 645 92 22];
            app.SelectedImageLabel.Text = 'Selected Image:';

            % Create EditedImageLabel
            app.EditedImageLabel = uilabel(app.UIFigure);
            app.EditedImageLabel.Position = [427 348 79 22];
            app.EditedImageLabel.Text = 'Edited Image:';

            % Create KernelSizeEditFieldLabel
            app.KernelSizeEditFieldLabel = uilabel(app.UIFigure);
            app.KernelSizeEditFieldLabel.Position = [27 389 66 22];
            app.KernelSizeEditFieldLabel.Text = 'Kernel Size';

            % Create KernelSizeEditField
            app.KernelSizeEditField = uieditfield(app.UIFigure, 'numeric');
            app.KernelSizeEditField.Position = [146 389 100 22];

            % Create MaxWindowSizeEditFieldLabel
            app.MaxWindowSizeEditFieldLabel = uilabel(app.UIFigure);
            app.MaxWindowSizeEditFieldLabel.Position = [27 343 104 22];
            app.MaxWindowSizeEditFieldLabel.Text = 'Max. Window Size';

            % Create MaxWindowSizeEditField
            app.MaxWindowSizeEditField = uieditfield(app.UIFigure, 'numeric');
            app.MaxWindowSizeEditField.Position = [146 343 100 22];

            % Create StandartDeviationEditFieldLabel
            app.StandartDeviationEditFieldLabel = uilabel(app.UIFigure);
            app.StandartDeviationEditFieldLabel.Position = [27 305 104 22];
            app.StandartDeviationEditFieldLabel.Text = 'Standart Deviation';

            % Create StandartDeviationEditField
            app.StandartDeviationEditField = uieditfield(app.UIFigure, 'numeric');
            app.StandartDeviationEditField.Position = [146 305 100 22];

            % Create Image
            app.Image = uiimage(app.UIFigure);
            app.Image.Position = [427 401 364 224];

            % Create Image2
            app.Image2 = uiimage(app.UIFigure);
            app.Image2.Position = [427 105 364 222];

            % Create SaveImageButton
            app.SaveImageButton = uibutton(app.UIFigure, 'push');
            app.SaveImageButton.ButtonPushedFcn = createCallbackFcn(app, @SaveImageButtonPushed, true);
            app.SaveImageButton.Position = [614 55 100 23];
            app.SaveImageButton.Text = 'Save Image';

            % Create ProcessButton
            app.ProcessButton = uibutton(app.UIFigure, 'push');
            app.ProcessButton.ButtonPushedFcn = createCallbackFcn(app, @ProcessButtonPushed, true);
            app.ProcessButton.Position = [228 473 100 23];
            app.ProcessButton.Text = 'Process';

            % Create AmplitudeEditFieldLabel
            app.AmplitudeEditFieldLabel = uilabel(app.UIFigure);
            app.AmplitudeEditFieldLabel.Position = [27 268 58 22];
            app.AmplitudeEditFieldLabel.Text = 'Amplitude';

            % Create AmplitudeEditField
            app.AmplitudeEditField = uieditfield(app.UIFigure, 'numeric');
            app.AmplitudeEditField.Position = [146 268 100 22];

            % Create MeanEditFieldLabel
            app.MeanEditFieldLabel = uilabel(app.UIFigure);
            app.MeanEditFieldLabel.Position = [27 235 38 22];
            app.MeanEditFieldLabel.Text = 'Mean ';

            % Create MeanEditField
            app.MeanEditField = uieditfield(app.UIFigure, 'numeric');
            app.MeanEditField.Position = [146 235 100 22];

            % Create ScaleEditFieldLabel
            app.ScaleEditFieldLabel = uilabel(app.UIFigure);
            app.ScaleEditFieldLabel.Position = [27 120 45 22];
            app.ScaleEditFieldLabel.Text = 'Scale';

            % Create ScaleEditField
            app.ScaleEditField = uieditfield(app.UIFigure, 'numeric');
            app.ScaleEditField.Position = [146 120 100 22];

            % Create ExponentialRateEditFieldLabel
            app.ExponentialRateEditFieldLabel = uilabel(app.UIFigure);
            app.ExponentialRateEditFieldLabel.Position = [27 84 96 22];
            app.ExponentialRateEditFieldLabel.Text = 'Exponential Rate';

            % Create ExponentialRateEditField
            app.ExponentialRateEditField = uieditfield(app.UIFigure, 'numeric');
            app.ExponentialRateEditField.Position = [146 84 100 22];

            % Create ErlangShapeEditFieldLabel
            app.ErlangShapeEditFieldLabel = uilabel(app.UIFigure);
            app.ErlangShapeEditFieldLabel.Position = [27 51 78 22];
            app.ErlangShapeEditFieldLabel.Text = 'Erlang Shape';

            % Create ErlangShapeEditField
            app.ErlangShapeEditField = uieditfield(app.UIFigure, 'numeric');
            app.ErlangShapeEditField.Position = [146 51 100 22];

            % Create ErlangRateEditFieldLabel
            app.ErlangRateEditFieldLabel = uilabel(app.UIFigure);
            app.ErlangRateEditFieldLabel.Position = [27 13 68 22];
            app.ErlangRateEditFieldLabel.Text = 'Erlang Rate';

            % Create ErlangRateEditField
            app.ErlangRateEditField = uieditfield(app.UIFigure, 'numeric');
            app.ErlangRateEditField.Position = [146 13 100 22];

            % Create SaltValueEditFieldLabel
            app.SaltValueEditFieldLabel = uilabel(app.UIFigure);
            app.SaltValueEditFieldLabel.Position = [23 197 59 22];
            app.SaltValueEditFieldLabel.Text = 'Salt Value';

            % Create SaltValueEditField
            app.SaltValueEditField = uieditfield(app.UIFigure, 'numeric');
            app.SaltValueEditField.Position = [146 197 100 22];

            % Create PepperValueEditFieldLabel
            app.PepperValueEditFieldLabel = uilabel(app.UIFigure);
            app.PepperValueEditFieldLabel.Position = [23 161 77 22];
            app.PepperValueEditFieldLabel.Text = 'Pepper Value';

            % Create PepperValueEditField
            app.PepperValueEditField = uieditfield(app.UIFigure, 'numeric');
            app.PepperValueEditField.Position = [146 161 100 22];

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = imageProcess

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end