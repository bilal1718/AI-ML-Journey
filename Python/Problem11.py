def add_expense(expenses, input_str):
    try:
        amount, category = input_str.split()
        amount = float(amount)
        expenses.append((amount, category))
    except ValueError:
        print("Invalid input. Please enter in the format 'amount category'.")
def calculate_category_totals(expenses):
    category_totals = {}
    for amount, category in expenses:
        if category in category_totals:
            category_totals[category] += amount
        else:
            category_totals[category] = amount
    return category_totals
def calculate_total(expenses):
    return sum(amount for amount, category in expenses)
def main():
    expenses = []
    while True:
        user_input = input("Enter an expense (amount category) or 'stop' to finish: ")
        if user_input.lower() == 'stop':
            break
        add_expense(expenses, user_input)
    category_totals = calculate_category_totals(expenses)
    total_expenses = calculate_total(expenses)
    print("\nExpense Report:")
    for category, total in category_totals.items():
        print(f"Total for {category}: ${total:.2f}")
    print(f"Overall total: ${total_expenses:.2f}")

main()
