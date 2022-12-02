import tkinter
import PySimpleGUI as sg
from model_perceptron import model_perceptron as perceptron
import numpy as np

class Screen:
    def __init__(self):
        self.flag = 2
        self.flag2 = 0
        self.cont = 0
        self.x_aux = []

    def evaluate_model_screen(self, list_evaluate_model, title, resp):
        sg.theme('DarkBlue')
    
        evaluate = [
            [sg.Text('')],
            [sg.Text(title)],
            [sg.Text(resp)],
            [sg.Text('')],
            [sg.Text('------------------------------------------------------------------------------------------')],
            [sg.Text('                            Model Evaluation Metrics')],
            [sg.Text('------------------------------------------------------------------------------------------')],
            [sg.Text('')],
            [sg.Text(f'- Accuracy score: {list_evaluate_model[0]}')],
            [sg.Text(f'- Error rate: {list_evaluate_model[1]}')],
            [sg.Text(f'- Zero One loss: {list_evaluate_model[2]}')],
            [sg.Text(f'- F1 scores:')],
            [sg.Text(f'- average macro: {list_evaluate_model[3]}')],
            [sg.Text(f'- average weighted: {list_evaluate_model[4]}')],
            [sg.Text(f'- average micro: {list_evaluate_model[5]}')],
            [sg.Text(f'- average binary: {list_evaluate_model[6]}')],
            [sg.Text(f'- Roc auc score: {list_evaluate_model[7]}')],
            [sg.Text(f'- Precision score: {list_evaluate_model[8]}')],
            [sg.Text(f'- Recall score: {list_evaluate_model[9]}')],
            [sg.Text(f'- Jaccard score: {list_evaluate_model[10]}')],
            [sg.Text(f'- Hamming loss score: {list_evaluate_model[12]}')],
            [sg.Text(f'- Max error: {list_evaluate_model[13]}')],
            [sg.Text(f'- Matthews correlation coefficient: {list_evaluate_model[14]}')],
            [sg.Text(f'- Matrix confusion:')],
            [sg.Text(list_evaluate_model[11])]
        ]

        retrun_evaluate = [[sg.Frame('Analysis Output', layout=evaluate, key='container_model_ealuation_metrics')]], 

        return sg.Window('Analysis Output', layout=retrun_evaluate, finalize=True)
    
    def initial_screen(self):
        sg.theme('DarkBlue')

        group = [
            [sg.Text('Álvaro Breno Prudêncio Brandão, Flávio Henrique Madureira Bergamini, Káryn Talyne dos Santos Silva e')],
            [sg.Text('Sávio Gomes Leite')],
        ]

        configuration1 = [
            [sg.Text('Input age (discrete values in the range 1 to 9):')],
            [sg.Input(key='age')],

            [sg.Text('Input menopause (discrete values in the range 1 to 3):')],
            [sg.Input(key='menopause')],

            [sg.Text('Input tumor_size (discrete values in the range 1 to 12):')],
            [sg.Input(key='tumor_size')],

            [sg.Text('Input inv_nodes (discrete values in the range 1 to 13):')],
            [sg.Input(key='inv_nodes')],

            [sg.Text('Input node_caps (discrete values in the range 1 to 2):')],
            [sg.Input(key='node_caps')],

            [sg.Text('Input deg_malig (discrete values in the range 1 to 3):')],
            [sg.Input(key='deg_malig')],

        ]
        
        configuration2 = [
            
            [sg.Text('Input breast (discrete values in the range 1 to 2):')],
            [sg.Input(key='breast')],

            [sg.Text('Input breast_quad (discrete values in the range 1 to 5):')],
            [sg.Input(key='breast_quad')],

            [sg.Text('Input irradiat (discrete values in the range 1 to 2):')],
            [sg.Input(key='irradiat')],

            

            [sg.Text('')],

            [sg.Button(' Predict now ')]
        ]

        screen = [
            [sg.Frame('Group', layout=group, key='container_initial_option')], 
            
            [sg.Frame('Configuration', layout=configuration1, key='container_configuration1'), 
            sg.Frame('Configuration', layout=configuration2, key='container_configuration1'),],
        ]

        layout = [ [sg.Frame('Inatel - Tópicos Especiais 2', layout=screen, key='container')]]

        window = sg.Window('C318', layout)

        while True:
            events, values = window.read()
            
            if events == sg.WIN_CLOSED: 
                window.close()
                break

            if events == ' Predict now ':
                self.cont = self.cont + 1
                
                checkX = False

                if values['age'] == '':
                    tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 9")
                    checkX = False
                else:
                    age = int(values['age'])
                    if(age < 1 or age > 9):
                        tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 9")
                    else:
                        self.x_aux.append(age)
                        checkX = True
                        

                if values['menopause'] == '':
                    tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 3")
                    checkX = False
                else:
                    menopause = int(values['menopause'])
                    if (menopause < 1 or menopause > 3):
                        tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 3")
                    else:
                        self.x_aux.append(menopause)
                        checkX = True

                if values['tumor_size'] == '':
                    tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 12")
                    checkX = False
                else:
                    tumor_size = int(values['tumor_size'])
                    if (tumor_size < 1 or tumor_size > 12):
                        tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 12")
                    else:
                        self.x_aux.append(tumor_size)
                        checkX = True

                if values['inv_nodes'] == '':
                    tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 13")
                    checkX = False
                else:
                    inv_nodes = int(values['inv_nodes'])
                    if (inv_nodes < 1 or inv_nodes > 13):
                        tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 13")
                    else:
                        self.x_aux.append(inv_nodes)
                        checkX = True

                if values['node_caps'] == '':    
                    tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 2")
                    checkX = False
                else:
                    node_caps = int(values['node_caps'])
                    if (node_caps < 1 or node_caps > 2):
                        tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 2")
                    else:
                        self.x_aux.append(node_caps)
                        checkX = True

                if values['deg_malig'] == '':
                    tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 3")
                    checkX = False
                else:
                    deg_malig = int(values['deg_malig'])
                    if (deg_malig < 1 or deg_malig > 3):
                        tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 3")
                    else:
                        self.x_aux.append(deg_malig)
                        checkX = True

                if values['breast'] == '':
                    tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 2")
                    checkX = False
                else:
                    breast = int(values['breast'])
                    if (breast < 1 or breast > 2):
                        tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 2")
                    else:
                        self.x_aux.append(breast)
                        checkX = True

                if values['breast_quad'] == '':
                    tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 5")
                    checkX = False
                else:
                    breast_quad = int(values['breast_quad'])
                    if (breast_quad < 1 or breast_quad > 5):
                        tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 5")
                    else:
                        self.x_aux.append(breast_quad)
                        checkX = True

                if values['irradiat'] == '':
                    tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 2")
                    checkX = False
                else:
                    irradiat = int(values['irradiat'])
                    if (irradiat < 1 or irradiat > 2):
                        tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 2")
                    else:
                        self.x_aux.append(irradiat)
                        checkX = True

                if checkX == True:
                    aux_x = (np.array(self.x_aux).reshape(-1,9))
                    data_x = aux_x
                    print("data_x -> ", data_x)

                    prt = perceptron()
                    model = prt.get_model()
                    pred = model.predict(data_x)
                    print("pred result -> ", pred)
                    title = '       >> Based on the analysis made by the model:'
                    resp = None
                    for b in pred:
                        if b==1:
                            resp = "    No, there is no great chance of recurrence of the disease."
                        elif b==2:
                            resp = "    Yes, there is a high chance of recurrence of the disease."
                    self.evaluate_model_screen(prt.evaluate_model(), title, resp)
                else:
                    tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value in field")

            if events == sg.WIN_CLOSED: 
                break
        window.close()
