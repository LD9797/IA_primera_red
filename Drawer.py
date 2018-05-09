import turtle
from mnist_loader import load_data


def see_picture(color):
    main_color = "black"
    adjuster = 255
    if color == 1:
        main_color = "white"
        adjuster = 0
    pictures_numbers, va_d, te_d = load_data()
    line_break_counter = 0
    turtle.colormode(255)
    turtle.speed(0)
    turtle.up()
    turtle.left(90)
    turtle.forward(270)
    turtle.right(90)
    turtle.down()
    for picture in pictures_numbers[0]:
        picture = 255 * (1.0 - picture)
        for pixel in picture:
            if pixel == 255.0:
                turtle.pencolor(main_color)
            else:
                turtle.pencolor(adjuster - int(pixel), adjuster - int(pixel), adjuster - int(pixel))
            turtle.forward(1)
            line_break_counter += 1
            if line_break_counter == 28:
                turtle.up()
                turtle.backward(28)
                turtle.right(90)
                turtle.forward(1)
                turtle.down()
                turtle.left(90)
                line_break_counter = 0


see_picture(0)
see_picture(1)
