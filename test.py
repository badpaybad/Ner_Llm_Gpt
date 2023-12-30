listGroupByVal=["grpc1","grpc1","grpcA","grpcB","grpcA","grpc1","grpcC",]

mapVal={}
idx=1
for val in listGroupByVal:
    if val in mapVal.keys():
        pass
    else:
        mapVal[val]=idx
        idx=idx+1

print(mapVal)

listIdxForVal=[]
for val in listGroupByVal:
    listIdxForVal.append(mapVal[val])
    
print(listGroupByVal)
print(listIdxForVal)