# Goals
1. Auto Tensorization  
  addressed by double forms
2. Machine Learning Friendly  
  addressed by Imperative Scheduling
3. Recursive Call/Intrinsics  
  addressed by callable block
4. Partial Tile  
  (partition, predicate, mask)
  UNADDRESSED
5. Unified Graph IR and Tensor IR  
  UNADDRESSED

# Data Structures

- Block  
    A Block marks a compute region. Every block has two forms:
    - **Form 1 Oqapue Form**: The expanded, general form. Use iteration vars (for loops) + stmt to represent computation. This form can be used to represent any halide programs. This form is also used for final code generation.
    - **Form 2 Compute Decleration Form**: The compact form. Use a single expression `decl` to represent computation in the block. The lhs of the `decl` only contains direct index 
    (e.g. can be `C[i, j]` but cannot be `C[io\*8+ii, j])`. The `decl` is used to match intrinsic declaration for hardware backends.
       Besides, we also need to store the mapping from the buffers in `decl` to real buffers. (This is necessary to reconstruct the block by using Form 2 only).

- ComputeDecl  
    Simplified tvm.compute expression. TC like expression (`C[i][j] +=! A[i][k] * B[k][j]`)

- TensorView  
    Bidirectional mapping between two tensor regions (`A[i][j] <-> sub_A[j],  0 <= i <= 16`)

- IndexRule  
    Bidirectional mapping for index   (`p <-> p + 12`)

# Examples

### Halide IR -> Tensor IR
We build blocks from Halide IR, then merge blocks and add compute declaration annotations.

- Matmul with partial tiles  
  ***Halide IR***
  ```python
  for (jo, 0, 12) {
    for (io, 0, 12) {
      for (ji.init, 0, 8) {
        for (ii.init, 0, 8) {
          C(((io*8) + ii.init), ((jo*8) + ji.init)) =0.000000f
        }
      }
      for (ko, 0, 16) {
        for (ji, 0, 8) {
          for (ii, 0, 8) {
            for (ki, 0, 8) {
              C(((io*8) + ii), ((jo*8) + ji)) =(C(((io*8) + ii), ((jo*8) + ji)) + (A(((io*8) + ii), ((ko*8) + ki))*B(((jo*8) + ji), ((ko*8) + ki))))
            }
          }
        }
      }
    }
    for (ji.init, 0, 8) {
      for (ii.init, 0, 4) {
        C((ii.init + 96), ((jo*8) + ji.init)) =0.000000f
      }
    }
    for (ko, 0, 16) {
      for (ji, 0, 8) {
        for (ii, 0, 4) {
          for (ki, 0, 8) {
            C((ii + 96), ((jo*8) + ji)) =(C((ii + 96), ((jo*8) + ji)) + (A((ii + 96), ((ko*8) + ki))*B(((jo*8) + ji), ((ko*8) + ki))))
          }
        }
      }
    }
  }
  for (io, 0, 12) {
    for (ji.init, 0, 4) {
      for (ii.init, 0, 8) {
        C(((io*8) + ii.init), (ji.init + 96)) =0.000000f
      }
    }
    for (ko, 0, 16) {
      for (ji, 0, 4) {
        for (ii, 0, 8) {
          for (ki, 0, 8) {
            C(((io*8) + ii), (ji + 96)) =(C(((io*8) + ii), (ji + 96)) + (A(((io*8) + ii), ((ko*8) + ki))*B((ji + 96), ((ko*8) + ki))))
          }
        }
      }
    }
  }
  for (ji.init, 0, 4) {
    for (ii.init, 0, 4) {
      C((ii.init + 96), (ji.init + 96)) =0.000000f
    }
  }
  for (ko, 0, 16) {
    for (ji, 0, 4) {
      for (ii, 0, 4) {
        for (ki, 0, 8) {
          C((ii + 96), (ji + 96)) =(C((ii + 96), (ji + 96)) + (A((ii + 96), ((ko*8) + ki))*B((ji + 96), ((ko*8) + ki))))
        }
      }
    }
  }
  ```
  
  ***Tensor IR***
  
  ```go
  Block ()                                                     (100, 100), (100,) Z[p, q] +=! X[p, r] * Y[q, r] : Z -> C, p -> p, q -> q, r -> r 
      Block (jo [0:12])                                        (96, 96), (100,),  Z[p, q] +=! X[p, r] * Y[q, r] : Z -> C, p -> p, q -> q, r -> r
          Block (io[0:12])                                     (96, 8), (100,),   Z[p, q] +=! X[p, r] * Y[q, r] : Z -> C, p -> p, q -> jo * 8 + q, r -> r
              Block ()                                         (8, 8), (),        Z[p, q] +=! X[p, r] * Y[q, r] : Z -> C, p -> io * 8 + p, q -> jo * 8 + q, r -> r
                  Block (ji.init[0:8], ii.init[0:8])           (8, 8), (),        Z[p, q] = 0 : Z -> C, p -> io * 8 + p, q -> jo * 8 + q
                      Block ()                                 (1, 1), (1, 1),    Z[p, q] = 0 : Z -> C, p -> io * 8 + ii.init, q -> jo * 8 + ji.init
                          C(((io*8) + ii.init), ((jo*8) + ji.init)) =0.000000f
                  Block (ko[0:16], ji[0:8], ii[0:8], ki[0:8])  (8, 8), (128,),    Z[p, q] += X[p, r] * Y[q, r]
                      Block ()                                 (1, 1), (1, 1),    Z[p, q] += X[p, r0 * 8 + r1] * Y[q, r0 * 8 + r1]
                          C(((io*8) + ii), ((jo*8) + ji)) =(C(((io*8) + ii), ((jo*8) + ji)) + (A(((io*8) + ii), ((ko*8) + ki))*B(((jo*8) + ji), ((ko*8) + ki))))
      //Block (jo [0:12])                                        (4, 96), (100,),   Z[p, q] +=! X[p, r] * Y[q, r] : Z -> C, p -> 96 + p, q -> q, r -> r
          Block ()                                             (4, 8), (100,),    Z[p, q] +=! X[p, r] * Y[q, r] : Z -> C, p -> 96 + p, q -> jo * 8 + q, r -> r
              Block (ji.init[0:8], ii.init[0:4])               (4, 8), (),        Z[p, q] = 0
                  Block ()                                     (1, 1), (1, 1),    Z[p, q] = 0
                      C((ii.init + 96), ((jo*8) + ji.init)) =0.000000f
              Block (ko[0:16], ji[0:8], ii[0:4], ki[0:8])      (4, 8), (128,),    Z[p, q] += X[p, r] * Y[q, r]
                  Block ()                                     (1, 1), (1, 1),    Z[p, q] += X[p, r0 * 8 + r1] * Y[q, r0 * 8 + r1]
                      C((ii + 96), ((jo*8) + ji)) =(C((ii + 96), ((jo*8) + ji)) + (A((ii + 96), ((ko*8) + ki))*B(((jo*8) + ji), ((ko*8) + ki))))
      Block (io[0:12])                                     (96, 4), (),     Z[p, q] +=! X[p, r] * Y[q, r] : Z -> C, p -> 96 + p, q -> 96 + q, r -> r
          Block()                                          (8, 4),  (),     Z[p, q] +=! X[p, r] * Y[q, r] : Z -> C, p -> io * 8 + p, q -> 96 + q, r -> r     
              Block (ji.init[0:4], ii.init[0:8])           (8, 4),  (),     Z[p, q] = 0  : Z -> C, p -> p, q -> q
                  Block ()                                 (1, 1),  (1, 1), Z[p, q] = 0  : Z -> C, p -> ii.init + 96, q -> ji.init + 96
                    C(((io*8) + ii.init), (ji.init + 96)) =0.000000f
              Block (ko[0:16], ji[0:4], ii[0:8], ki[0:8])  (4, 8), (128,),  Z[p, q] += X[p, r] * Y[q, r] :  Z -> C, p -> p, q -> 96 + q, r0 -> r0, r1 -> r1
                  Block ()                                 (1, 1), (1, 1),  Z[p, q] += X[p, r0 * 8 + r1] * Y[q, r0 * 8 + r1] :  Z -> C, p -> io*8 + ii, q -> ji + 96, r0 -> ko, r1 -> ki
                      C(((io*8) + ii), (ji + 96)) =(C(((io*8) + ii), (ji + 96)) + (A(((io*8) + ii), ((ko*8) + ki))*B((ji + 96), ((ko*8) + ki))))
      Block ()                                             (4, 4), (),      Z[p, q] +=! X[p, r] * Y[q, r] : Z -> C, p -> 96 + p, q -> 96 + q, r -> r
          Block (ji.init[0:4], ii.init[0:4])               (4, 4), (),      Z[p, q] = 0 : Z -> C, p -> 96, q -> 96
              Block ()                                     (1, 1), (1, 1),  Z[p, q] = 0 : Z -> C, p -> ii.init + 96, q -> ji.init + 96 
                  C((ii.init + 96), (ji.init + 96)) =0.000000f
          Block (ko[0:16], ji[0:4], ii[0:4], ki[0:8])      (4, 4), (128,),  Z[p, q] += X[p, r] * Y[q, r] : Z -> C, p -> 96 + p, q -> 96 + q, r -> r
              Block ()                                     (1, 1), (1, 1),  Z[p, q] += X[p, r0 * 8 + r1] * Y[q, r0 * 8 + r1] :  Z -> C, p -> ii + 96, q -> ji + 96, r0 -> ko, r1 -> ki
                  C((ii + 96), (ji + 96)) =(C((ii + 96), (ji + 96)) + (A((ii + 96), ((ko*8) + ki))*B((ji + 96), ((ko*8) + ki))))
  ```
  
- Strided innermost access  
  **Halide IR**
  ```python
  for (im, 0, 4) {
    for (io, 0, 16) {
      for (ii, 0, 2) {
        C((((io*8) + (im*2)) + ii)) = ((((io*8) + (im*2)) + ii) + 1
      }
    }
  }
  ```
  
  **Tensor IR**
  ```go
  Block          (im[0:4])   (128,), (),  Z[p] = p * p : p -> (p % 32)/2*8 + p/32 * 2 + p % 2
      Block      (io[0:16])  (32,),  (),  Z[p] = p * p : p -> p/2 * 8 + im * 2 + p % 2
          Block  (ii[0:2])   (2,),   (),  Z[p] = p * p : p -> io * 8 + im * 2 + p
              Block  ()      (1,),   (),  Z[p] = p * p : p -> io * 8 + im * 2 + ii
                   C((((io*8) + (im*2)) + ii)) = (io*8 + im*2 + ii)*(io*8 + im*2 + ii)
  ```

- With compute\_at  
  **Halide IR**
  ```python
  for (i, 0, 16) {
    for (j, 0, 16) {
      for (ii, 0, 3) {
        B((i + ii), j) =(((A((i + ii), j) + A((i + ii), (j + 1))) + A((i + ii), (j + 2)))/3.000000f)
      }
      C(i, j) =(((B(i, j) + B((i + 1), j)) + B((i + 2), j))/3.000000f)
    }
  }
  ```

  **Tensor IR**
  ```go
  Block               (i[0:16])
      Block           (j[0:16])    !! How to merge B and C ??
          Block       (i[0:3])     (3,), (),  Z[p, q] = (X[p, q] + X[p, q+1] + X[p, q+2]) / 3.0 : Z- > B, X -> A, p -> i + p, q -> j
              Block ()             (1,), (),  Z[p, q] = (X[p, q] + X[p, q+1] + X[p, q+2]) / 3.0 : Z- > B, X -> A, p -> i + ii, q -> j
                  B((i + ii), j) =(((A((i + ii), j) + A((i + ii), (j + 1))) + A((i + ii), (j + 2)))/3.000000f)
          Block()                  (1,), (),  Z[p, q] = (X[p, q] + X[p+1, q] + X[p+2, q]) / 3.0 : Z -> C, X -> B, p -> i, q -> j
              C(i, j) =(((B(i, j) + B((i + 1), j)) + B((i + 2), j))/3.000000f)
  ```

- Max, Argmax
  - Max  
    ```go
    A_red() = -inf
    for (k0, 0, 10) {
      for (k1, 0, 10) {
        A_red() = max(A_red(), A(k0, k1))
      }
    }
    ```

    ```
    Block        ()              (), (10, 10),  Z[] max=! X[r0, r1] : Z -> A_red, X -> A, r0 -> r0, r1 -> r1
        Block    ()              (), (),        Z[] = -inf
        Block    (k0[0:10])      (), (10, 10),  Z[] max= X[r0, r1] : Z -> A_red, X -> A, r0 -> r0, r1 -> r1
            Block    (k1[0:10])  (), (1, 10),   Z[] max= X[r0, r1] : Z -> A_red, X -> A, r0 -> k0, r1 -> r1
                Block ()         (), (1, 1),    Z[] max= X[r0, r1] : Z -> A_red, X -> A, r0 -> k0, r1 -> k1
                    A_red() = max(A_red(), A(k0, k1))
    ```

  - Argmax
    ```python
    for (ax0, 0, 10) {
      A_red_temp.value[0](ax0) = -1
      A_red_temp.value[1](ax0) = -340282346638528859811704183484516925440.000000f
      for (k0, 0, 10) {
        A_red_temp.value[0](ax0) = select((A(k0, ax0) <= A_red_temp.value[1](ax0)), A_red_temp.value[0](ax0), k0)
        A_red_temp.value[1](ax0) = select((A(k0, ax0) <= A_red_temp.value[1](ax0)), A_red_temp.value[1](ax0), A(k0, ax0))
      }
    }
    for (ax0, 0, 10) {
      A_red(ax0) = A_red_temp(ax0)
    }
    ```

    ```go
    Block (ax0[0:10])                   !! How to merge? extract identity + fcombine?
        Block ()                        (1,), (),    Z.0[p] = -1, Z.1[p] = -inf : Z.0 -> A_red_tmp.v[0], p -> ax0
            A_red_temp(ax0).value[0] = -1
            A_red_temp(ax0).value[1] = -340282346638528859811704183484516925440.000000f
        Block (ko[0:10])                (1,), (10,), Z.0[p] = select(X[r, p] <= Z.1[p], Z.0[p], r), Z.0[p] = select(X[r, p] <= Z.1[p], Z.1[p], X[r, p]) : Z.0 -> A_red_tmp.v[0], p -> ax0, r -> r
            Block()                     (1,), (1,),  Z.0[p] = select(X[r, p] <= Z.1[p], Z.0[p], r), Z.0[p] = select(X[r, p] <= Z.1[p], Z.1[p], X[r, p]) : Z.0 -> A_red_tmp.v[0], p -> ax0, r -> k0
                A_red_temp.value[0](ax0) = select((A(k0, ax0) <= A_red_temp.value[1](ax0)), A_red_temp.value[0](ax0), k0)
                A_red_temp.value[1](ax0) = select((A(k0, ax0) <= A_red_temp.value[1](ax0)), A_red_temp.value[1](ax0), A(k0, ax0))
    Block (ax0[0:10])
        A_red(ax0) = A_red_temp(ax0)
    ```

    Need to support:
      1. convert multiple statements in the innermost as multiple outputs op

- Scan


- Irregular Access  
  - Case 1 (Correct, equivalent to fuse)  
      **Halide IR**
      ```python
      for (io, 0, 16) {
          for (ii, 0, 8) {
              C(io + ii) = io + ii
          }
      }
      ```
      
      **Tensor IR**
      ```go
      Block (io[0:16])        (128,), (), Z[p] = p : p -> p / 8 + p % 8
          Block (ii[0:8])     (8,),   (), Z[p] = p : p -> io + p
               Block ()       (1,),   (), Z[p] = p : p -> io + ii
                    C(io + ii) = io + ii
      ```
  - Case 2 (Introduce extra computation)  
    **Halide IR**
    ```python
    for (io, 0, 16) {
        for (ii, 0, 8) {
            C(io*8 + ii, io*8 + ii) = io * 8 + ii
        }
    }
    ```
    
    **Tensor IR**
    ```go
    Block (io[0:16])        (128,128), (),  Z[p, q] = p : p -> p, q -> q
        Block (ii[0:8])     (8,8),     (),  Z[p, q] = p : p -> io*8 + p,  q -> io*8 + q
            Block ()        (1,1),     (),  Z[p, q] = p : p -> io*8 + ii, q -> io*8 + ii
                C(io*8 + ii, io*8 + ii) = io * 8 + ii
    ```



### Tensorization
We can view a block without free vars as an intrinsic

1. Use dot to compose a gemm
  Tensor IR

  ```go
  # dot_intrin
  Block ()         (), (16,),  Z += X[r] * Y[r]  : Z -> C, X -> A, Y -> B, r -> r
      tvm.call_packed("dot", addr_of(C[0]), addr_of(A[0]), addr_of(B[0]))

  # gemm
  Block (i[0:128], j[0:128])         (128, 128), (128,),   Z[p, q] +=! X[p, r] * Y[q, r] : Z -> C, p -> p, q -> q
      Block ()                       (128, 128), (),       Z[p, q] = 0
          C(i, j) =0.000000f
      Block (ko[0:16])               (1, 1), (128,),       Z[p, q] += X[p, r] * Y[q, r] : Z -> C, p -> i, q -> j, r -> r
          Block (ki[0:8])            (1, 1), (1, 8),       Z[p, q] += X[p, r] * Y[q, r] : Z -> C, p -> i, q -> j, r -> ko * 8 + r
              Block ()               (1, 1), (1, 1),       Z[p, q] += X[p, r] * Y[q, r] : Z -> C, p -> i, q -> j, r -> ko * 8 + ki
                  C(i, j) = (C(i, j) + (A(i, ((ko*8) + ki))*B(j, ((ko*8) + k.inner))))
  ```

  -> replace 

  ```go
  Block (i[0:128], j[0:128])         (128, 128), (128,),   Z[p, q] +=! X[p, r] * Y[q, r]
      Block ()                       (128, 128), (),       Z[p, q] = 0
          C(i, j) =0.000000f
      Block (ko[0:16])               (1, 1), (128,),       Z[p, q] += X[p, r] * Y[q, r] : Z -> C, p -> i, q -> j, r -> r
          tvm.call_packed('dot', addr_of(C[i][j]), addr_of(A[i][ko * 8]), addr_of(A[j][ko * 8]))
  ```

  -> register as a new intrinsic

  Need to support:
    1. fuzzy matching
    2. relative index with different dimensions

2. Use gemv to compose a gemm

3. Use fixed-length intrinsic to compose a general-length intrinsic
   ```go
   # vmul_intrin:
   Block ()                 (16,), Z[p] = X[p] * Y[p]: Z -> D, p -> p
       tvm.call_packed("vmul", addr_of(D[0]), addr_of(E[0]), addr_of(F[0]))

   # vmul_general_block:
   Block (i[0:n])           (n,),  Z[p] = X[p] * Y[p]: Z -> C, p -> p
       Block()              (1,),  Z[p] = X[p] * Y[p]: Z -> C, p -> i
           C[i] = A[i] * B[i]
   ```

   -> split, factor=16

   ```go
   Block ()                            (n,),        Z[p] = X[p] * Y[p]: Z -> C, p -> p
       Block (io[0:n//16])             (n//16*16,), Z[p] = X[p] * Y[p]: Z -> C, p -> p
           Block (ii[0:16])            (16,),       Z[p] = X[p] * Y[p]: Z -> C, p -> io * 16 + p
               Block()                 (1,),        Z[p] = X[p] * Y[p]: Z -> C, p -> io * 16 + ii
                   C[io*16 + ii] = A[io * 16 + ii] * B[io*16 + ii]
       Block (ii[0:n%16])              (n % 16,),   Z[p] = X[p] * Y[p]: Z -> C, p -> p
           Block()                     (1,),        Z[p] = X[p] * Y[p]: Z -> C, p -> n//16*16 + ii
               C[n//16*16 + ii] = A[n//16*16 + ii] * B[n//16*16 + ii]
   ```

   -> tensorize with vmul_intrin

   ```go
   Block ()                            (n,),        Z[p] = X[p] * Y[p]: Z -> C, p -> p
       Block (io[0:n//16])             (n//16*16,), Z[p] = X[p] * Y[p]: Z -> C, p -> p
           Block ()                    (16,),       Z[p] = X[p] * Y[p]: Z -> C, p -> io * 16 + p
               tvm.call_packed("vmul", addr_of(C[io * 16]), addr_of(A[io * 16]), addr_of(B[io * 16]))
       Block (ii[0:n%16])              (n % 16,),   Z[p] = X[p] * Y[p]: Z -> C, p -> p
           Block()                     (1,),        Z[p] = X[p] * Y[p]: Z -> C, p -> n//16*16 + ii
               C[n//16*16 + ii] = A[n//16*16 + ii] * B[n//16*16 + ii]
   ```

   -> register as a new intrin


5. Matching bypass (Tensor op, Callable Blocks)
   Use opaque expression in `decl`. It only marks dependency relation. It can be matched to any other `decl`.

   ```go
   # vmul_intrin:
   Block#0 ()                 (16,), Z[p] = X[p]: Z -> D, p -> p
       tvm.call_packed("unexpressable", addr_of(D[0]), addr_of(E[0]))

   # vmul_general_block:
   Block#1 (io[0:n/16])         (n/16,), Z[p] = OPAQUE(X[p]) : Z -> C, p -> p
       Block#2 (ii[0:16])       (16,),   Z[p] = OPAQUE(X[p]) : Z -> C, p -> io * 16 + p
           Block#3 ()           (1,),    Z[p] = OPAQUE(X[p]) : Z -> C, p -> i
               C[p] = OPAQUE(A[p])
   ```

   -> 

   ```
   Block#1 (io[0:n/16])         (n/16,), Z[p] = OPAQUE(X[p]) : Z -> C, p -> p
       tvm.call_packed("unexpressable", addr_of(C[0]), addr_of(A[0]))
   ```

6. Absolute index / Relative index
  ??


### Split
```go
   Block (i[0:n])           (n,),  Z[p] = X[p] * Y[p]: Z -> C, p -> p
       Block()              (1,),  Z[p] = X[p] * Y[p]: Z -> C, p -> i
           C[i] = A[i] * B[i]
```

->

```go
   Block ()                            (n,),        Z[p] = X[p] * Y[p]: Z -> C, p -> p
       Block (io[0:n//16])             (n//16*16,), Z[p] = X[p] * Y[p]: Z -> C, p -> p
           Block (ii[0:16])            (16,),       Z[p] = X[p] * Y[p]: Z -> C, p -> io * 16 + p
               Block()                 (1,),        Z[p] = X[p] * Y[p]: Z -> C, p -> io * 16 + ii
                   C[io*16 + ii] = A[io * 16 + ii] * B[io*16 + ii]
       Block (ii[0:n%16])              (n % 16,),   Z[p] = X[p] * Y[p]: Z -> C, p -> p
           Block()                     (1,),        Z[p] = X[p] * Y[p]: Z -> C, p -> n//16*16 + ii
               C[n//16*16 + ii] = A[n//16*16 + ii] * B[n//16*16 + ii]
```
1. add two blocks `perfect` and `partial`, copy children to these two blocks
2. replace vars in children block, do not need to update shape
3. update shape and `idx_rule` for newly built two blocks (`io = i / factor, ii = i % factor`)

### Fuse

### Reorder

### Compute At

### Reorganize axis in blocks
1. BO, BI = split(Block, outer\_axes, inner\_axes)
2. B = fuse(Block, BO, BI)
3. Reorder Blocks?
4. Reassign axes

Merge blocks, update `idx_rule` and shape

### Lower to Halide IR
Use the (for + stmt) form, should be easy.


# Alg

### Build From Halide IR
1. For inner most statement, build an empty block
    1. get statement
    2. **solve equation** to remove complex index in lhs
    3. build `decl`, `idx_rule`, `buf_rule`
2. Look up to gather a perfect loop nest
    1. update shape: multiply
    2. update `idx_rule`: for every new axis `i`, there should be only one rule `p` contains it. Replace it with
        -> old_p = p % lower_shape, i = p / lower_shape
    3. (?) fuse reduction : if the domain of reduction vars is known, try to fuse them
3. Merge blocks  
    Check index expression:  
    - init + aggregation  
        check shape and mapping are __the same__
    - same  
        check shape and mapping are __continuous__
    - other   
        cannot merge, split upper itervars to all children blocks

### Tensoirzation (Detect and Auto-transform)
1. sub-graph isomorphism problem?

### Tensoirzation (Matching and Replace)
1. Matching
    - match low dimension to high dimension 
    - 

2. Replace
    - replace buffer 
    - use bidirectional `idx_rule` to replace index (constraint on stride)

### Lower to Halide IR
Use the (for + stmt) form, should be easy.

# Operation API

# Utilities
1. Bidirectional expression function  (e.g.  `p <-> p + 12`)
2. quasi-affine equation solver
3. Tensor View (variable dimensions)

# Discussion
1. Where to put if / allocate? Inside a block or stmt?
2. How to introduce equation solver? isl or hand-written?
3. Dependency analysis??
4. absolute index / relative index in tensorization?
5. Why imperative scheduling is hard?

