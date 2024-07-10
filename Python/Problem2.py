import random, sys
random_num=random.randint(1,100)
user_points=10
print("Welcome to the Mysterious Number Game!")
print("You have 10 points to start with.")
print("Guess the mysterious number between 1 and 100.")
while user_points > 0:
    guess=int(input("Enter your guess: "))
    if guess < 1 or guess > 100:
        print("Guess should be between 1 and 100")
        break

    if guess==random_num:
        print(f"Congratulations! You've guessed the mysterious number {random_num}.")
        sys.exit()
    else:
        user_points=user_points - 1
        if guess < random_num:
                print("Too low! Try again.")
        else:
                print("Too high! Try again.")
        print(f"Points remaining: {user_points}")

print(f"Sorry, you've lost the game. The mysterious number was {random_num}.")
sys.exit()
