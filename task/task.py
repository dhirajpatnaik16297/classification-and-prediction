import tensorflow as tf
import numpy as np
import pandas as pd
import csv
#reading csv files and clearing
data = pd.read_csv("data_train.csv")
data.replace(np.nan,0.0,inplace=True)
data_test = pd.read_csv("data_test.csv")
data_test.replace(np.nan,0.0,inplace=True)
features=['num1','num2','num3','num4','num5','num6','num7',	'num8','num9','num10','num11','num12','num13',
          'num14','num15','num16','num17','num18','num19','num20','num21','num22','num23','der1','der2',
          'der3','der4','der5','der6','der7','der8','der9','der10','der11','der12','der13','der14','der15',
          'der16','der17','der18','der19','cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9',
          'cat10','cat11','cat12','cat13','cat14']
target='target'
# the datas can also bet normalised
X_train = data[features]
y_train = data[target]
X_test = data_test[features]

print("Dimensions of the training set : {0}".format(np.shape(X_train)))
print("Dimensions of the training set (target) : {0}".format(np.shape(y_train.values.reshape(len(y_train),1))))

feature_columns = [tf.feature_column.numeric_column("x", shape=[len(features),1])]

#classifier using estimators
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[64,128,64],n_classes=2,
                                        optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                                        activation_fn=tf.nn.relu)
#training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.array(X_train)},
    y=np.array(y_train.values.reshape((len(y_train),1))),
    num_epochs=None,
    shuffle=True)
#testing inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.array(X_test)},
    num_epochs=1,
    shuffle=False)
#training the model
classifier.train(input_fn=train_input_fn, steps=10000)

#prediction of probabilities 
predict_results = list(classifier.predict(input_fn=test_input_fn))

#saving the probabilities and ids of data_test.csv to output.csv
test_id_out=[]
test_prob_1=[]
for idx, prediction in enumerate(predict_results[:1000]):
    test_id_out.append(test_ids[idx])
    test_prob_1.append(prediction['probabilities'][1])

with open('output.csv','w') as out:
        csv.writer(out).writerows(zip(["ids"],["probability for 1"]))
        csv.writer(out, quoting=csv.QUOTE_MINIMAL).writerows(zip(test_id_out, test_prob_1))
