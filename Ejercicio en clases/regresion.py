


vector=[2,0,1,3]#puede tener cualquier tamaño n
n=4
p=2#siempre tendra valor n/2
val=[]
for i in range(p):
    val.append(vector)
 
indice=0
for i in range(p):
    for j in range(n):
        if j==indice:
            aux=val[i][j]
            val[i][j]=val[i][n-1]
            val[i][n-1]=aux
    indice=indice +1
 
print (val)

