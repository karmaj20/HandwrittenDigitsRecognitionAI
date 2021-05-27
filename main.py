import GUI
import GUIScratch
from GUI import *
from GUIScratch import *

def main():
    print('Welcome to the program that recognizes handwritten digits')
    print('---------------------------------------------------------')
    print('Choose the neural network model that suits you better')
    print('--------Menu--------')
    print('Choose D if you prefer Dense model with 2 hidden layers')
    print('Choose C if you prefer model build on Convolutional ')
    print('Choose S if you prefer model build from scratch ')


    choice = input('Your choice: ')
    if choice == 'D':
        model = GUI.Window(choice)
    elif choice == 'C':
        model = GUI.Window(choice)
    elif choice == 'S':
        model = GUIScratch.Window()
    else:
        print('Wrong choice')


if __name__ == '__main__':
    main()