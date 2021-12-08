import pickle
import sys

obj = list()
f = open(sys.argv[1], "rb")

i = 0
while True:
    try:
        tmp = pickle.load(f)
        # pickle.dump(tmp, sys.stdout.buffer)
        # break
        obj.append(tmp)
        if i % 15 == 0:
            with open(f"{i}.data", "wb") as fd:
                pickle.dump(tmp, fd)
    except EOFError as e:
        break
    i += 1
f.close()
print(obj)