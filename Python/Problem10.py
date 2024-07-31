def Decoder(message):
    message_decode=""
    for number in message:
        if 1 <= number <= 26:
         message_decode+=chr(number + 64)
    return message_decode
secret_code=[20, 8, 5, 18, 5]
decode=Decoder(secret_code)
print(decode)

