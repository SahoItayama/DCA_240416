filename = "parameters_txt/parameters40-100-2000a.txt"

with open(filename, "r") as file:
    lines = file.readlines()[5:9]

for line in lines:
    print(line.strip())

print(type)