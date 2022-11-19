import numpy as np
import matplotlib.pyplot as plt
from _data import ImportSQL
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import hamming_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import max_error


class model_perceptron:
    
    def __init__(self):
        None
        
    def get_model(self):
        
        self.x_treino, self.x_teste, self.y_treino, self.y_teste = ImportSQL.GetData()
        self.model = MLPClassifier(max_iter=10000, tol=1e-3, random_state=1)
        self.model.fit(self.x_treino, self.y_treino)

        return self.model

        
    def evaluate_model(self):

        y_pred = self.model.predict(self.x_teste)

        #positive class: there is recurrence of the disease (target = 2)
        initial_proba = self.model.predict_proba(self.x_teste)
        local_list = []
        for i in range(56):
            local_list.append(initial_proba[i][1])

        y_scores = np.array(local_list)

        #Accuracy
        accuracy = accuracy_score(self.y_teste, y_pred, normalize=True)

        #Error rate
        error_rate = 1 - accuracy

        #Compute the F1 scores, also known as balanced F-score or F-measure.

        f1_macro = f1_score(self.y_teste, y_pred, average='macro')
        f1_weighted = f1_score(self.y_teste, y_pred, average='weighted')
        f1_micro = f1_score(self.y_teste, y_pred, average='micro')
        f1_binary = f1_score(self.y_teste, y_pred, average='binary')

        #Compute the precision

        Precision = precision_score(self.y_teste, y_pred, average='weighted')

        #Compute the recall

        r_score = recall_score(self.y_teste, y_pred, average='weighted')

        # Compute Area Under the Curve (AUC) from prediction scores
        roc_auc_score_value = roc_auc_score(self.y_teste, y_scores)

        #Jaccard similarity coefficient score

        J_score = jaccard_score(self.y_teste, y_pred, average='weighted')

        #Confusion matrix(Accuracy by computing the confusion matrix with each row corresponding to the true class)

        #The hamming_loss computes the average Hamming loss or Hamming distance between two sets of samples.
        h_loss = hamming_loss(self.y_teste, y_pred)

        #The matthews_corrcoef function computes the Matthew’s correlation coefficient (MCC) for binary classes
        m_coef = matthews_corrcoef(self.y_teste, y_pred)

        #Max error(computes the maximum residual error)
        M_error = max_error(self.y_teste, y_pred)

        # Zero-one classification loss (The best performance is 0)
        zero_one_loss_value = zero_one_loss(self.y_teste, y_pred)

        c_matrix = confusion_matrix(self.y_teste, y_pred, labels=self.model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=self.model.classes_)
        disp.plot()
        #plt.savefig('src/resources/confusion_matrix1.png', format='png')
        plt.show()

        # Printing metrics
        '''
        print("\n-------------------------------------------------------------------------------------------------")
        print("                                      MODEL EVALUATION METRICS")
        print("-------------------------------------------------------------------------------------------------")
        print("\n - Accuracy score: ", round(accuracy, 2))
        print("\n - Error rate: ", round(error_rate, 2))
        print("\n - Zero One loss: ", round(zero_one_loss_value, 2))
        print("\n - F1 scores: ")
        print("     average macro: ", round(f1_macro, 2))
        print("     average weighted: ", round(f1_weighted, 2))
        print("     average micro: ", round(f1_micro, 2))
        print("     average binary: ", round(f1_binary, 2))
        print("\n - Roc auc score: ", round(roc_auc_score_value, 2))
        print("\n - Precision score: ", round(Precision, 2))
        print("\n - Recall score: ", round(r_score, 2))
        print("\n - Jaccard score: ", round(J_score, 2))
        print("\n - Matrix confusion: \n", c_matrix)
        print("\n - Hamming loss score: ", round(h_loss, 2))
        print("\n - Max error: ", round(M_error, 2))
        print("\n - Matthews correlation coefficient: ", round(m_coef, 2))
        print("-------------------------------------------------------------------------------------------------\n") '''

        return [round(accuracy, 2), round(error_rate, 2), round(zero_one_loss_value, 2), round(f1_macro, 2), round(f1_weighted, 2),
        round(f1_micro, 2), round(f1_binary, 2), round(roc_auc_score_value, 2), round(Precision, 2), round(r_score, 2), round(J_score, 2), 
        c_matrix, round(h_loss, 2), round(M_error, 2), round(m_coef, 2)]
        