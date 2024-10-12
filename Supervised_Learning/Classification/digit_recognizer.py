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


##################################################

# ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds=roc_curve(y_train_5, y_scores)


idx_for_threshold_at_90=(thresholds <= threshold_for_90_precision).argmax()

tpr_90, fpr_90=tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.plot(fpr, tpr, linewidth=2, label="ROC Curve")
plt.plot([0,1], [0,1], 'k:', label="Random Classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
[...]
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)

from sklearn.ensemble import RandomForestClassifier
forest_clf=RandomForestClassifier(random_state=42)

y_probas_forest=cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_forest=y_probas_forest[:,1]
precisions_forest, recalls_forest, thresholds_forest=precision_recall_curve(y_train_5, y_scores_forest)

plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2, label="Random Forest")

plt.plot(recall, precisions, "--", linewidth=2, label="SGD")
[...]
plt.show()

y_train_pred_forest=y_probas_forest[:,1] >= 0.5
f1_score(y_train_5, y_train_pred_forest)
roc_auc_score(y_train_5, y_scores_forest)


#################################################

# MultiClass Classification

from sklearn.svm import SVC

svm_clf=SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])
svm_clf.predict([some_digit])

some_digit_scores=svm_clf.decision_function([some_digit])
some_digit_scores.round(2)

class_id=some_digit_scores.argmax()

from sklearn.multiclass import OneVsRestClassifier

ovr_clf=OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])
ovr_clf.predict([some_digit])
