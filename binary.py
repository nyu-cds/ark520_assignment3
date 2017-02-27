import itertools
    
def zbits(n,k):
    """produces all n-digit binary numbers with k zeros in it.
    """
    all_ones = ["1"] * n
    to_return = set()
    for where_zeros_are in itertools.combinations(range(n), k):
        digits = all_ones.copy()
        for pos in where_zeros_are:
            digits[pos] = "0"
        to_return.add( "".join(digits) )
    return to_return