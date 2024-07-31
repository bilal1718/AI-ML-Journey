def hidden_message(file):
    hidden_message_chars = []

    with open(file, 'r') as openFile:
        for line in openFile:
            parts = line.split(maxsplit=3)
            if len(parts) > 3:
                message = parts[3].replace(" ", "")
                if len(message) >= 5:
                    hidden_message_chars.append(message[4])
    return ''.join(hidden_message_chars)
hidden_message_output = hidden_message("logs.txt")
print("Hidden message:", hidden_message_output)
