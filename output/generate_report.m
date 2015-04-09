function generate_report()
%GENERATE_REPORT Generates the mean error and accuracy reports
%   Example: generate_report()

  % Load inter-occular distance error and compute each facial-feature
  % mean error and accuracy
  error = load('errors.txt');
  avg = mean(error);
  INTER_OCCULAR_DISTANCE = 0.1;
  accuracy = (sum(error<INTER_OCCULAR_DISTANCE)/size(error, 1))*100;

  % Comparison to paper results
  x = {'left eye (l)', 'left eye (r)', 'mouth (l)', 'mouth (r)', ...
       'mouth (b)', 'mouth (t)', 'right eye (l)', 'right eye (r)', ...
       'nose (l)', 'nose (r)'};
     
  avg_paper = [0.068 0.056 0.073 0.078 0.095 0.064 0.056 0.073 0.059 0.07];
  bar([avg' avg_paper'])
  set(gca, 'XTickLabel', x)
  xlabel('Facial Feature');
  ylabel('Inter-Occular Distance');
  legend('binary','paper');
  
  accuracy_paper = [87.7 93.5 81.9 80.8 71.5 86.7 92.9 86.2 90.4 88.2];
  figure, bar([accuracy' accuracy_paper'])
  set(gca, 'XTickLabel', x)
  xlabel('Facial Feature');
  ylabel('Accuracy');
  legend('binary','paper');
end