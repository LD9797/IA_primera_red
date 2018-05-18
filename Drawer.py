import turtle
from mnist_loader import load_data
import pickle


def read():
    pickle_in = open("C:\\Users\\User\\Desktop\\Resultados\\Pesos256.pkl", "rb")
    example_dict = pickle.load(pickle_in)
    biases = pickle.load(open("C:\\Users\\User\\Desktop\\Resultados\\Biases256.pkl", "rb"))
    return example_dict


def see_picture(color):
    main_color = "black"
    adjuster = 255
    if color == 1:
        main_color = "white"
        adjuster = 0
    va_d, pictures_numbers = read()
    line_break_counter = 0
    turtle.colormode(255)
    turtle.speed(0)
    turtle.up()
    turtle.left(90)
    turtle.forward(270)
    turtle.right(90)
    turtle.down()
    image = 0
    for picture in pictures_numbers:
        image += 1
        picture = 255 * (1.0 - abs(picture))
        for pixel in picture:
            if pixel == 255.0:
                turtle.pencolor(main_color)
            else:
                try:
                    turtle.pencolor(abs(adjuster - abs(int(pixel))), abs(adjuster - abs(int(pixel))), abs(adjuster - abs(int(pixel))))
                except:
                    pass
            turtle.forward(1)
            line_break_counter += 1
            if line_break_counter == 16:
                turtle.up()
                turtle.backward(16)
                turtle.right(90)
                turtle.forward(1)
                turtle.down()
                turtle.left(90)
                line_break_counter = 0
        turtle.up()
        turtle.right(90)
        turtle.forward(10)
        turtle.down()
        turtle.left(90)

        if image == 11:
            turtle.up()
            turtle.forward(35)
            turtle.left(90)
            turtle.forward(16*9 + 10*9)
            turtle.right(90)
            turtle.down()
            image = 0
    input()


see_picture(0)
