def Highest_Purchaser(purchase):
    highest_spender_id = ""
    high_price = 0
    for customer_id, items in purchase.items():
        total_price = 0
        for item in items:
            total_price += item["quantity"] * item["price"]
        if total_price > high_price:
            high_price = total_price
            highest_spender_id = customer_id
    return highest_spender_id, high_price

purchases = {
    "C001": [
        {"item": "laptop", "quantity": 1, "price": 1200},
        {"item": "mouse", "quantity": 2, "price": 20}
    ],
    "C002": [
        {"item": "phone", "quantity": 1, "price": 800},
        {"item": "headphones", "quantity": 1, "price": 150}
    ],
    "C003": [
        {"item": "monitor", "quantity": 2, "price": 300},
        {"item": "keyboard", "quantity": 1, "price": 100}
    ]
}

result = Highest_Purchaser(purchases)
print(result)
