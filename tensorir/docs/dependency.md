# Dependnecy Conditions

1. Elemwise defined dependency
2. Only depends on the last statement writes to it


for i in range(1, 10):
    for j in range(1, 10):
        A[i][j] = 32

for i in range(1, 10):
    for j in range(1, 10):
        A[i][j] = 123
        B[i][j] = A[i][j+1]


## Inline dependency 

inline statement X   W(b), R(a), R(c)

1. set middle W to W(b), expand to both sides
2. set middle R to R(a,c), expand to both sides
3. remove symbols near '+'
4. replace R(b) in right with R(a,c)

*
RAW + RAW = W R + W(b) R(b)
          = W(a) R(a) + W(b) R(b) or W(c) R(c) + W(b) R(b)  | no
          = W(a) R(b) or W(c) R(b)
          = W(a) R(a,c) or W(c) R(a,c)                      | RAW

RAW + WAW = W R + W(b) W(b)                                 
          = W(a) R(a) + W(b) W(b) or W(c) R(c) + W(b) W(b)  | no
          = W(a) W(b) or W(c) W(b)                          | no

*
WAW + WAW = W(b) W(b) + W(b) W(b)                           | no
          = W(b) W(b)                                       | WAW

WAW + RAW = W(b) W(b) + W(b) R(b)                           | no
          = W(b) R(b)                               
          = W(b) R(a,c)                                     | no

RAW + WAR = W R + R W = W R(a or c) + R(a or c) W
             1. W(a) R(a) + R(a) W(a) = W(a) W(a) | WAW -> WAW
          =  2. W(a) R(a) + R(c) W(c) = W(a) W(c) | no  -> no
             3. W(c) R(c) + R(a) W(a) = W(c) W(a) | no  -> no
             4. W(c) R(c) + R(c) W(c) = W(c) W(c) | WAW -> WAW

WAW + WAR = W W + R W = W(b) W(b) + R W
                      = W(b) W(b) + R(a) W(a) or W(b) W(b) + R(c) W(c)  | no or no
                      = W(b) W(a) or W(b) W(c)                          | no or no


WAR + WAR = R W + R W = R(b) W(b) + R W
                      = R(b) W(b) + R(a) W(a) or R(b) W(b) + R(c) W(c) | no or no
                      = R(b) W(a) or R(b) W(c)                         | no or no

*
WAR + WAW = R W + W W = R(b) W(b) + W(b) W(b) | no
                      = R(b) W(b)             | WAR      

WAW + RAW = W W + W R = R(b) W(b) + W(b) R(b) | no 
                      = R(b) R(a,c) | no

RAR + RAW = R R W(b) R(b)
          = R(a) R(a) W(b) R(b) or R(c) R(c) W(b) R(b) | no or no
          = R(a) R(a,c) or  R(c) R(a, c)               | RAR or RAR

RAR + WAW = R R W(b) W(b)
          = R(a) R(a) W(b) W(b) | no
          = R(a) W(b)           | no

RAR + WAR = R R R W  -> keep constant

RAR + RAR = R R R R  -> keep constant

RAW + RAR = W R R R  -> keep constant

WAR + RAR = R(b) W(b) R R
          = R(b) W(b) R(a) R(a) or R(b) W(b) R(c) R(c) | no or no
          = R(b) R(a) or R(b) R(c)                     | no or no

WAW + RAR = W(b) W(b) R R
          = | no or no
          = | no or no

