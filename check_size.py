import os

sizebox = []
for filename in os.listdir('text'):
    f = open('text/' + filename, 'r')
    size = os.path.getsize(f.name)
    sizebox.append(size)
    print(f.name, size, 'B')

print('average:', sum(sizebox)/len(sizebox), 'B')
print('max:', max(sizebox), 'B')
print('min:', min(sizebox), 'B')