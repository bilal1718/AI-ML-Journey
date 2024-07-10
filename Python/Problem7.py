import pyperclip
pyperclip.copy("Lists of animals\nLists of aquarium life\nLists of biologists by author\nabbreviation\nLists of cultivars")
text = pyperclip.paste()

lines = text.split('\n')
for i in range(len(lines)):
    lines[i] = '* ' + lines[i]
text = '\n'.join(lines)
pyperclip.copy(text)
print(text)