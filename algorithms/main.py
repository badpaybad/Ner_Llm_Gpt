def findMax(arr):
    temp = 0
    tempidx = 0
    for idx, val in enumerate(arr):
        if val > temp:
            temp = val
            tempidx = idx

    return (tempidx, temp)


def subArray(arr, fromIdx, toIdx):
    return arr[fromIdx:toIdx]


def findNextEdgeIndex(arr, fromEdgeIdx, fromEdgeVal):
    j = fromEdgeIdx+1
    length = len(arr)
    subarr = subArray(arr, j, length)
    # print(f"subarr: {subarr} val: {val} idx: {idx}")
    b = j
    for jsub, jval in enumerate(subarr):
        if jval >= fromEdgeVal:
            b = b + jsub
            break
    # print(f"b: {b}")
    if b == j:
        jsub, jval = findMax(subarr)
        b = b+jsub
        # print(f"bMax: {b} jsub: {jsub}")
    return (b, subarr)


def findHoles(arr):
    holes = []
    length = len(arr)
    idx = 0
    while True:
        if idx == length-1:
            break
        val = arr[idx]
        j = idx+1
        if val <= arr[j]:
            idx = idx+1
            continue
        a = idx
        (b, subarr) = findNextEdgeIndex(arr, idx, val)
        if b == j:
            break
        idx = b
        b=b+1
        hole = subArray(arr, a, b)
        # print(f"hole: {hole} b: {b}")        
        holes.append((hole, a, b))
    return holes


def calHoleValue(arrHole):
    a = arrHole[0]
    b = arrHole[len(arrHole)-1]
    if a > b:
        a = b
    d = 0
    for c in arrHole:
        val = a-c
        if val > 0:
            d = d+val
    return d


arr = [1, 2, 7, 3, 2, 1, 7, 2, 4, 6, 7, 8, 9, 9, 2, 5, 6, 7, 4, 5, 3, 2, 1]
holes = findHoles(arr)

for h, i, j in holes:
    print(f"hole found from {i}-{j}: {h} hole val: {calHoleValue(h)}")
