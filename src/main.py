from _perceptron import model_perceptron as perceptron
import numpy as np
from _screen import Screen

def main():
    flag = int(input(">> Hi!Would you like to classify arbitrary data? If yes, input 0 ; if not, input 2\n"))
    flag2 = 0
    cont = 0
    x = []
    x_aux = []
    y = []

    # Aqui, o usuário insere valores que são usados para a predição (os novos dados), utilizando-se do modelo já treinado.
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
        #Aqui, é imformado o resultado da predição. 
        data_x = np.array(x_aux).reshape(-1,9)
        data_y = np.array(y)
        print(">>Based on the data, there is a recurrence of the disease:\n")
        prt = perceptron.model_perceptron()
        model = prt.get_model()
        pred = model.predict(data_x)
        print("Result:\n", pred.reshape(-1,1), "\n")
        for b in pred:
            if b==1:

                print(b," -> NO\n")
            elif b==2:
                print(b," -> YES")
        print("\n\n")        
        prt.evaluate_model()
        
        

if __name__ == '__main__': 
    Tela = Screen.Screen()
    Tela.initial_screen()
    
    # main() 