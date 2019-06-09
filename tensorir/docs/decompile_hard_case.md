# Hardcases for decompile

## non-affine expressions

1. fuse spatial and reduction : -> ????
```
C(i, (j.k.fused/128)) =(C(i, (j.k.fused/128)) + (A(i, (j.k.fused % 128))*B((j.k.fused/128), (j.k.fused % 128))))
-> solve equation
C[p, q] = C[p, q] + A[p, q * 128 % 128] * B[q, q * 128 % 128]
```

2. fuse spatial
```
C(((i.outer*8) + (j.outer.i.inner.fused % 8)), (((j.outer.i.inner.fused/8)*8) + j.inner)) =(C(((i.outer*8) + (j.outer.i.inner.fused % 8)), (((j.outer.i.inner.fused/8)*8) + j.inner)) + (A(((i.outer*8) + (j.outer.i.inner.fused % 8)), k)*B((((j.outer.i.inner.fused/8)*8) + j.inner), k)))
-> solve equation
C(i, j) = C(i, j) + (A(i, k)*B(j, k))

C(((i.outer*8) + (j.outer.i.inner.fused % 8)), (((j.outer.i.inner.fused/8)*8) + j.inner)) =(C(((i.outer*8) + (j.outer.i.inner.fused % 8)), (((j.outer.i.inner.fused/8)*8) + j.inner)) + (A(((i.outer*8) + (j.outer.i.inner.fused % 8)), k)*B((((i.outer*8) + j.outer.i.inner.fused) + j.inner), k)))
-> solve equation
C(i, j) = C(i, j) + (A(i, k)*B(i + j, k))    # isl can solve this
```

3. split reduction             : -> add a compress reduction stage
```
C(((i.outer*8) + i.inner), (j.inner + 96)) =(C(((i.outer*8) + i.inner), (j.inner + 96)) + (A(((i.outer*8) + i.inner), ((k.outer*10) + k.inner))*B((j.inner + 96), ((k.outer*10) + k.inner))))
-> solve equation
C[p, q] = C[p, q] + A[p, ko * 10 + ki] * B[q, ko * 10 + ki]
```

4. layout transform           :
The belowing example is easy. Some may need to use BijectiveLayout transformation.
```
C(i, j) = (C(i, j) + (A(i, k)*B((j/8), k, (j % 8))))
->
C[p, q] = C[p, q] + A[p, k] * B[p/8, k, p % 8]
```

## other cases
1. compute\_at
2. multiple outputs op

