from GUI import *

def main():
    print('Welcome to the program that recognizes handwritten digits')
    print('---------------------------------------------------------')
    print('Choose the neural network model that suits you better')
    print('--------Menu--------')
    print('Choose D if you prefer Dense model with 2 hidden layers')
    print('Choose C if you prefer model build on Convolutional ')


    choice = input('Your choice: ')
    if (choice == 'D'):
        model = Window(choice)
    elif (choice == 'C'):
        model = Window(choice)
    else:
        print('Wrong choice')


if __name__ == '__main__':
    main()