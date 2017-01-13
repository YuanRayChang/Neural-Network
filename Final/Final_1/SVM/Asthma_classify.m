function result = Asthma_classify(TestSample)
load('SVM_model.mat', 'SVMModel');
result = predict(SVMModel,TestSample);
end