import pygame
import sys
import numpy as np

letter_index = np.array([])
letter_array = np.array(["h","e","l","o","w","r","d"])

# Initialize pygame
pygame.init()
trial_lenght = 10;

# init timer
TIMER = pygame.USEREVENT + 1
pygame.time.set_timer(TIMER, trial_lenght * 1000)

# Set up display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Display Input Letter")

# Set colors
white = (255, 255, 255)
black = (0, 0, 0)

# Set font
font = pygame.font.Font(None, 200)  # Adjust the size of the font as needed

# Variable to store the letter
letter = ""

# Main loop
running = True
restcounter = 0
array_index = 0

while running:
    screen.fill(white)  # Fill the screen with white

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # elif event.type == pygame.KEYDOWN:
        #     # Get the letter from the key press and store it
        #     letter = event.unicode.upper()  # Display in uppercase for clarity
        #     letter_index = np.append(letter_index, letter)
        #     print(letter_index)
        
        elif event.type == TIMER:
            print("timer loop arrived")
            if (restcounter % 2 == 0):
                #rest
                screen.fill(white)
                letter = ""
                print("letter arrived")
            else:
                screen.fill(white)
                print("letter arrived")
                letter = letter_array[restcounter//12]

            restcounter += 1



            




    # Render the letter to the screen
    if letter:
        text_surface = font.render(letter, True, black)
        text_rect = text_surface.get_rect(center=(width // 2, height // 2))
        screen.blit(text_surface, text_rect)

    pygame.display.flip()

# Quit pygame
pygame.quit()
sys.exit()
