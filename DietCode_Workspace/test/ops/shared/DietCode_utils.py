def cross_product(argA, argB):
    args = []
    for a in argA:
        for b in argB:
            if isinstance(a, tuple):
                args.append((*a, *b)) if isinstance(b, tuple) else args.append((*a, b))
            else:
                args.append((a,  *b)) if isinstance(b, tuple) else args.append((a,  b))
    return args
