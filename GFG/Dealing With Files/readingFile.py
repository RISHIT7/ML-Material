lines = ""
with open('data.txt', 'r') as file:
    lines = file.read()
file.close()

print(lines)
