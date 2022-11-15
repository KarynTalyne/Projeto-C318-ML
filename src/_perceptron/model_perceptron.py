import numpy as np
import matplotlib.pyplot as plt
from _data import ImportSQL
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import hamming_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

class model_perceptron:
    
    def __init__(self):
        None
        
    def get_model(self):
        
        self.x_treino, self.x_teste, self.y_treino, self.y_teste = ImportSQL.GetData()
        self.model = Perceptron(max_iter=10000, tol=1e-3, random_state=1)
        self.model.fit(self.x_treino, self.y_treino)

        return self.model

        
    def evaluate_model(self):

        y_pred = self.model.predict(self.x_teste)

        #Accuracy
        accuracy = accuracy_score(self.y_teste, y_pred, normalize=True)*100

        # Average precision from prediction scores
        acc_v = []
        for i in range(self.y_teste.size):
             y_pred1 = self.model.predict(self.x_teste)

             accuracy = accuracy_score(self.y_teste, y_pred1, normalize=True) * 100
             acc_v.append(float(accuracy))

        accuracy_values = np.array(acc_v)
        avg_precision_scores = average_precision_score(self.y_teste, accuracy_values)*100

        # probabilities based on the confidence score of the samples, using the hyperplane distance
        mprob = (self.model.decision_function(self.x_teste)).tolist()

        for k in range(len(mprob)):
            if mprob[k] < 0:
                mprob[k] = -1 * (mprob[k])

        for j in range(len(mprob)):
            mprob[j] = mprob[j] / (max(mprob))

            model_prob = np.array(mprob)

            #Compute the Brier score loss(The smaller the Brier score loss, the better)
            brier_sc = brier_score_loss(self.y_teste, np.array(model_prob))*100

            #Compute the F1 scores, also known as balanced F-score or F-measure.

            f1_macro = f1_score(self.y_teste, y_pred, average='macro')*100
            f1_weighted = f1_score(self.y_teste, y_pred, average='weighted')*100

            #Log loss, aka logistic loss or cross-entropy loss.

            Log_loss = log_loss(self.y_teste, model_prob)

            #Compute the precision

            Precision = precision_score(self.y_teste, y_pred, average='weighted')*100

            #Compute the recall

            r_score = recall_score(self.y_teste, y_pred, average='weighted')*100

            #Jaccard similarity coefficient score

            J_score = jaccard_score(self.y_teste, y_pred, average='weighted')*100

            #Confusion matrix(Accuracy by computing the confusion matrix with each row corresponding to the true class)

            c_matrix = confusion_matrix(self.y_teste, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=self.model.classes_)
            disp.plot()
            plt.show()

            #The hamming_loss computes the average Hamming loss or Hamming distance between two sets of samples.
            h_loss = hamming_loss(self.y_teste, y_pred)*100

            #The matthews_corrcoef function computes the Matthew’s correlation coefficient (MCC) for binary classes
            m_coef = matthews_corrcoef(self.y_teste, y_pred)

            #Explained variance score

            ev_score = explained_variance_score(self.y_teste, y_pred)

            #Max error(computes the maximum residual error)
            M_error = max_error(self.y_teste, y_pred)

            #Mean absolute error
            Mean_error = mean_absolute_error(self.y_teste, y_pred)

            #Mean absolute percentage error
            Mean_P_error = mean_absolute_percentage_error(self.y_teste, y_pred)

            # Printing metrics

            print("\n-------------------------------------------------------------------------------------------------")
            print("                                      MODEL EVALUATION METRICS")
            print("-------------------------------------------------------------------------------------------------")
            print("\n - Accuracy score: ", round(accuracy, 2), "%")
            print("\n - Average precision from prediction score: ", round(avg_precision_scores, 2), "%")
            print("\n - Brier score loss: ", round(brier_sc, 2), "%")
            print("\n - F1 scores: ")
            print("     average macro: ", round(f1_macro, 2), "%")
            print("     average weighted: ", round(f1_weighted, 2), "%")
            print("\n - Log loss score: ", round(Log_loss, 2))
            print("\n - Precision score: ", round(Precision, 2), "%")
            print("\n - Recall score: ", round(r_score, 2), "%")
            print("\n - Jaccard score: ", round(J_score, 2), "%")
            print("\n - Matrix confusion: \n", c_matrix)
            print("\n - Hamming loss score: ", round(h_loss, 2), "%")
            print("\n - Explained variance score: ", round(ev_score, 2))
            print("\n - Max error: ", round(M_error, 2))
            print("\n - Mean absolute error score: ", round(Mean_error, 2))
            print("\n - Mean absolute percentage error score: ", round(Mean_P_error, 2), "%")
            print("-------------------------------------------------------------------------------------------------\n")