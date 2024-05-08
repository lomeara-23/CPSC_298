import random
import time
import os

# Define constants for board size, delay and direction values
BOARD_WIDTH = 20
BOARD_HEIGHT = 15
DELAY = 0.1

LEFT = (-1, 0)
RIGHT = (1, 0)
UP = (0, -1)
DOWN = (0, 1)

# Define a class for the snake game
class Snake:
    def __init__(self):
        self.snake_segments = [(4, 4), (3, 4), (2, 4)]
        self.direction = RIGHT
        self.score = 0
        self.food = (random.randint(1, BOARD_WIDTH - 2), random.randint(1, BOARD_HEIGHT - 2))

    def move(self):
        head = (self.snake_segments[0][0] + self.direction[0], self.snake_segments[0][1] + self.direction[1])
        if head in self.snake_segments or head[0] < 0 or head[0] >= BOARD_WIDTH or head[1] < 0 or head[1] >= BOARD_HEIGHT:
            return False  # Game over if snake collides with itself or goes out of bounds
        self.snake_segments.insert(0, head)
        if head == self.food:
            self.score += 1
            while True:
                new_food = (random.randint(1, BOARD_WIDTH - 2), random.randint(1, BOARD_HEIGHT - 2))
                if new_food not in self.snake_segments:
                    break
            self.food = new_food
        else:
            self.snake_segments.pop()
        return True

    def change_direction(self, direction):
        if (self.direction[0] * -1, self.direction[1] * -1) != direction:  # Prevent snake from moving in opposite direction
            self.direction = direction

# Function to draw the game board and score
def draw(snake_game):
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear console output for new frame
    print("Score:", snake_game.score)
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            if (x, y) in snake_game.snake_segments:
                print('X', end='')
            elif (x, y) == snake_game.food:
                print('O', end='')
            else:
                print(' ', end='')
        print()  # Move to next line
    time.sleep(DELAY)

# Main function to start the game
def main():
    snake_game = Snake()
    while True:
        draw(snake_game)
        if not snake_game.move():
            print("Game Over, Your Score is : ", snake_game.score)
            break

if __name__ == "__main__":
    main()

