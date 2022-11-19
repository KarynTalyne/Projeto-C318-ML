import tkinter
import PySimpleGUI as sg
from _perceptron import model_perceptron as perceptron
import numpy as np

class Screen:
    def __init__(self):
        self.flag = 2
        self.flag2 = 0
        self.cont = 0
        self.x = []
        self.x_aux = []
        self.y = []

    
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

            [sg.Text('Input breast (discrete values in the range 1 to 2):')],
            [sg.Input(key='breast')],
        ]
        
        configuration2 = [
            [sg.Text('Input breast_quad (discrete values in the range 1 to 5):')],
            [sg.Input(key='breast_quad')],

            [sg.Text('Input irradiat (discrete values in the range 1 to 2):')],
            [sg.Input(key='irradiat')],

            [sg.Text('Input class (discrete values in the range 1 to 2):')],
            [sg.Input(key='class_')],

            [sg.Text('Input deg_malig (discrete values in the range 1 to 3):')],
            [sg.Input(key='deg_malig')],

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
                checkY = False

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
                    self.x.append(self.x_aux)
                
                if values['class_'] == '':
                    tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 2")
                    checkY = False
                else:
                    class_ = int(values['class_'])
                    if (class_ < 1 or class_ > 2):
                        tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value between 1 and 2")
                    else:
                        self.y.append(class_)
                        checkY = True

                if checkX == True and checkY == True:
                    data_x = np.array(self.x_aux).reshape(-1,9)
                    data_y = np.array(self.y)


                    prt = perceptron.model_perceptron()
                    model = prt.get_model()
                    pred = model.predict(data_x)

                    tkinter.messagebox.showinfo(title="Based on the data, there is a recurrence of the disease", message=f'Result: {pred.reshape(-1,1)}')
                    # Obs: Nessa parte preferi deixar a apresentação via terminal, pois a quandidade de iterações são grandes e na interface acredito que não fica legal
                    for b in pred:
                        if b==1:

                            print(b," -> NO\n")
                        elif b==2:
                            print(b," -> YES")
                    print("\n\n")        
                    prt.evaluate_model()
                else:
                    tkinter.messagebox.showerror(title="WRONG INPUT!", message="Enter a value in field")

            if events == sg.WIN_CLOSED: 
                break
        window.close()
