import sys

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))
filename = sys.argv[1]

country_map = {}
with open(filename) as f:
    for line in f:
        content = line.strip()
        print(content)