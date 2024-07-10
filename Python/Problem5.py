def oldest_age(data):
    friend_age=0
    friend_name=""
    for person, info in data.items():
       for friend in info["friends"]:
        if friend in data:
          person_age= data[friend]["age"]
       if person_age > friend_age:
           friend_name=friend
           friend_age=person_age

    return friend_age, friend_name



data = {
    "Alice": {"age": 25, "friends": ["Bob", "Charlie"]},
    "Bob": {"age": 30, "friends": ["Alice"]},
    "Charlie": {"age": 35, "friends": ["Alice", "Bob"]}
}

result=oldest_age(data)
print(f"{result[1]} is the oldest with age {result[0]}")