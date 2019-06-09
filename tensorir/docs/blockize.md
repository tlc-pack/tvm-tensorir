# Blocksize

procedure:
1. check  (1) 
2. replace upper loop vars with arguments
3. deduce bound

## Examples

```python
for io = 0 to N / 8
    for jo = 0 to N / 8
        for ii = 0 to 8
            for ji = 0 to 8
                C[io * 8 + ii, jo * 8 + jj] = xxxxx
```

blockize ->

```python
for io = 0 to N / 8
    for jo = 0 to N / 8
        block(io, jo)  # block(X, Y)  C[X*8:X*8+8, Y*8:Y*8+8]
```

blocksize ->

```python
block()  # block()  C[0:N, 0:N]
```

### Compute\_at

#### Before blckize
```python
for io = 0 to N / 8
    for jo = 0 to N / 8
        for ii = 0 to 8
            for ji = 0 to 8
                C[io * 8 + ii, jo * 8 + jj] = xxxxx

for i = 0 to N
    for j = 0 to N
        D[i][j] = C[i][j]
```

C.compute\_at(D, i)

require C[i, 0:N]

for i = 0 to N
    for ax0 = i to i
        for ax1 = 0 to N
            C[ax0, ax1] = ...
    for j = 0 to N
        D[i][j] = C[i][j]

#### After blckize
```python
for io = 0 to N / 8
    for jo = 0 to N / 8
        block(io, jo)  # block(X, Y)  C[X*8:X*8+8, Y*8:Y*8+8]

for i = 0 to N
    for j = 0 to N
        D[i][j] = C[i][j]
```

C.compute\_at(D, i)

require C[i, 0:N]

```python
for i = 0 to N
    for ax0 = i to i
        for ax1 = 0 to N/8
            block(ax0, ax1)  # block(X, Y)  C[X*8:X*8+8, Y*8:Y*8+8]
    for j = 0 to N
        D[i][j] = C[i][j]
```

# Multi output

```python
for i = 0 to N
    for ax0 = i to i
        for ax1 = 0 to N/8
            block(ax0, ax1)  # block(X, Y)  C[X*8:X*8+8, Y*8:Y*8+8]
    for j = 0 to N
        D[i][j] = C[i][j]

for i = 0 to N
    for j = 0 to N
        E[i][j] = D[i][j]
```

blockize ->

```python
for i = 0 to N
    block(i)    # block(X) C[X*8,X*8+8, 0:N] D[X:X+1, 0:N]

for i = 0 to N
    for j = 0 to N
        E[i][j] = D[i][j]
```

compute\_at(E, i)

require D[i:i+1, 0:N]

for i = 0 to N
    for ax0 = i to i
        block(ax0)

    for j = 0 to N
        E[i][j] = D[i][j]

# Multi output



