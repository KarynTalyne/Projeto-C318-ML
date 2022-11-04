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


class Perceptron:

    # Load data
    x_treino, x_teste, y_treino, y_teste = ImportSQL.GetData()

    # Creating model

    model = Perceptron(max_iter=10000, tol=1e-3, random_state=1)

    # Fitting model

    model.fit(x_treino, y_treino)

    #Evaluating model

    y_pred = model.predict(x_teste)

    #Accuracy
    accuracy = accuracy_score(y_teste, y_pred, normalize=True)*100

    # Average precision from prediction scores
    acc_v = []
    for i in range(y_teste.size):

        y_pred1 = model.predict(x_teste)

        accuracy = accuracy_score(y_teste, y_pred1, normalize=True) * 100
        acc_v.append(float(accuracy))

    accuracy_values = np.array(acc_v)
    avg_precision_scores = average_precision_score(y_teste, accuracy_values)*100

    # probabilities based on the confidence score of the samples, using the hyperplane distance
    mprob = (model.decision_function(x_teste)).tolist()

    for k in range(len(mprob)):
        if mprob[k] < 0:
            mprob[k] = -1 * (mprob[k])

    for j in range(len(mprob)):
        mprob[j] = mprob[j] / (max(mprob))

    model_prob = np.array(mprob)

    #Compute the Brier score loss(The smaller the Brier score loss, the better)
    brier_sc = brier_score_loss(y_teste, np.array(model_prob))*100

    #Compute the F1 scores, also known as balanced F-score or F-measure.

    f1_macro = f1_score(y_teste, y_pred, average='macro')*100
    f1_weighted = f1_score(y_teste, y_pred, average='weighted')*100

    #Log loss, aka logistic loss or cross-entropy loss.

    Log_loss = log_loss(y_teste, model_prob)

    #Compute the precision

    Precision = precision_score(y_teste, y_pred, average='weighted')*100

    #Compute the recall

    r_score = recall_score(y_teste, y_pred, average='weighted')*100

    #Jaccard similarity coefficient score

    J_score = jaccard_score(y_teste, y_pred, average='weighted')*100

    #Confusion matrix(Accuracy by computing the confusion matrix with each row corresponding to the true class)

    c_matrix = confusion_matrix(y_teste, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=model.classes_)
    disp.plot()
    plt.show()

    #The hamming_loss computes the average Hamming loss or Hamming distance between two sets of samples.
    h_loss = hamming_loss(y_teste, y_pred)*100

    #The matthews_corrcoef function computes the Matthew’s correlation coefficient (MCC) for binary classes
    m_coef = matthews_corrcoef(y_teste, y_pred)

    #Explained variance score

    ev_score = explained_variance_score(y_teste, y_pred)

    #Max error(computes the maximum residual error)
    M_error = max_error(y_teste, y_pred)

    #Mean absolute error
    Mean_error = mean_absolute_error(y_teste, y_pred)

    #Mean absolute percentage error
    Mean_P_error = mean_absolute_percentage_error(y_teste, y_pred)

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

    flag = int(input(">> Hi!Would you like to classify arbitrary data? If yes, input 0 ; if not, input 2\n"))
    flag2 = 0
    cont = 0
    x = []
    x_aux = []
    y = []

    while flag != 1 and flag!=2:
        cont = cont + 1
        age = int(input("Input age (discrete values in the range 1 to 9):\n"))
        if(age<1 or age>9):
            print("WRONG INPUT! RESTART!")
            break
        x_aux.append(age)

        menopause = int(input("Input menopause (discrete values in the range 1 to 3):\n"))
        if (menopause < 1 or menopause > 3):
            print("WRONG INPUT! RESTART!")
            break
        x_aux.append(menopause)

        tumor_size = int(input("Input tumor_size (discrete values in the range 1 to 12):\n"))
        if (tumor_size < 1 or tumor_size > 12):
            print("WRONG INPUT! RESTART!")
            break
        x_aux.append(tumor_size)

        inv_nodes = int(input("Input inv_nodes (discrete values in the range 1 to 13):\n"))
        if (inv_nodes < 1 or inv_nodes > 13):
            print("WRONG INPUT! RESTART!")
            break
        x_aux.append(inv_nodes)

        node_caps = int(input("Input node_caps (discrete values in the range 1 to 2):\n"))
        if (node_caps < 1 or node_caps > 2):
            print("WRONG INPUT! RESTART!")
            break
        x_aux.append(node_caps)

        deg_malig = int(input("Input deg_malig (discrete values in the range 1 to 3):\n"))
        if (deg_malig < 1 or deg_malig > 3):
            print("WRONG INPUT! RESTART!")
            break
        x_aux.append(deg_malig)

        breast = int(input("Input breast (discrete values in the range 1 to 2):\n"))
        if (breast < 1 or breast > 2):
            print("WRONG INPUT! RESTART!")
            break
        x_aux.append(breast)

        breast_quad = int(input("Input breast_quad (discrete values in the range 1 to 5):\n"))
        if (breast_quad < 1 or breast_quad > 5):
            print("WRONG INPUT! RESTART!")
            break
        x_aux.append(breast_quad)

        irradiat = int(input("Input irradiat (discrete values in the range 1 to 2):\n"))
        if (irradiat < 1 or irradiat > 2):
            print("WRONG INPUT! RESTART!")
            break
        x_aux.append(irradiat)

        x.append(x_aux)

        class_ = int(input("Input class (discrete values in the range 1 to 2):\n"))
        if (class_ < 1 or class_ > 2):
            print("WRONG INPUT! RESTART!")
            break
        y.append(class_)

        flag = int(input("Predict now? if yes, input 1; if not, input 0\n"))
        print("\n\n")
        if(flag==1):
            flag2 = 1



    if flag2 == 1:

        data_x = np.array(x_aux).reshape(-1,9)
        data_y = np.array(y)
        print(">>Based on the data, there is a recurrence of the disease:\n")
        pred = model.predict(data_x)
        print("Result:\n", pred.reshape(-1,1), "\n")
        for b in pred:
            if b==1:

                print(b," -> NO\n")
            elif b==2:
                print(b," -> YES")
