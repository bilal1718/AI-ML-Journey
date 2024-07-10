def collatz(number):
    if number % 2 == 0:
        result = number // 2
    else:
        result = 3 * number + 1
    print(result)
    return result

try:
    value = int(input("Enter a number: "))
    while value != 1:
        value = collatz(value)
except ValueError:
    print("Invalid input. Please enter an integer.")
