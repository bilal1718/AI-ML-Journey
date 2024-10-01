from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


#################################

# MNIST Dataset
mnist=fetch_openml('mnist_784', as_frame=False)
X,y=mnist.data, mnist.target
print(X.shape)
print(y.shape)
def plot_digit(image_data):
    image=image_data.reshape(28,28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

some_digit=X[0]
plot_digit(some_digit)

X_train, X_test, y_train, y_test=X[:60000], X[60000:], y[:60000], y[60000:]

#####################################

# Training a Binary Classifier
y_train_5=(y_train=='5')
y_test_5=(y_test=='5')

from sklearn.linear_model import SGDClassifier

sdg_clf=SGDClassifier(random_state=42)
sdg_clf.fit(X_train, y_train_5)

sdg_clf.predict([some_digit])



########################################

# Measuring Accuracy using cross validation

from sklearn.model_selection import cross_val_score
cross_val_score(sdg_clf, X_train, y_train_5, cv=3, scoring="accuracy")

from sklearn.dummy import DummyClassifier

dummy_clf=DummyClassifier()
dummy_clf.fit(X_train, y_train_5)

cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")

############################################

# Confusion Matrices

from sklearn.model_selection import cross_val_predict

y_train_pred=cross_val_predict(sdg_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_train_5, y_train_pred)

y_train_perfect_predictions=y_train_5
confusion_matrix(y_train_5, y_train_perfect_predictions)


####################################################

# Precision and Recall

from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)

recall_score(y_train_5, y_train_pred)

from sklearn.metrics import f1_score
f1=f1_score(y_train_5, y_train_pred)
print(f1)

#######################################################

# Precision Recall Tradeoff

y_scores=sdg_clf.decision_function([some_digit])
print(y_scores)
y_scores=cross_val_predict(sdg_clf, X_train, y_train_5, cv=3, method="decision_function")

y_train_pred_90 = (y_scores > 70000)
precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)
from sklearn.metrics import precision_recall_curve
precisions, recall, thresholds=precision_recall_curve(y_train_5, y_scores)

idx_for_90_precision=(precisions >= 90).argmax()
threshold_for_90_precision=thresholds[idx_for_90_precision]

y_train_pred_90=(y_scores >= threshold_for_90_precision)

prec_scr=precision_score(y_train_5, y_train_pred_90)
print(prec_scr)
recall_at_90_precision=recall_score(y_train_5, y_train_pred_90)
print(recall_at_90_precision)

from sklearn.metrics import roc_curve
fpr, tpr, thresholds=roc_curve(y_train_5, y_scores)


idx_for_threshold_at_90=(thresholds <= threshold_for_90_precision).argmax()

tpr_90, fpr_90=tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]


