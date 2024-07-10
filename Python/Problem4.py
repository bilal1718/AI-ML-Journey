def CommaCode(listInput):
    if len(listInput)==0:
       return "Enter valid list"
    commaString=""
    for i in range(len(listInput)):
        if i == len(listInput)-1:
           commaString += " and " + listInput[i]
        elif i == len(listInput) -2:
           commaString+=listInput[i]
        else:
         commaString += listInput[i] + " , "

    return commaString

fruits = ['apples', 'bananas','mango', 'pomegrante']
empty=[]
emptyValue=CommaCode(empty)
stringValue=CommaCode(fruits)
print(emptyValue)
print(stringValue)