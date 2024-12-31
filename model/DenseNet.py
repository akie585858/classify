from mylib.module.Dense import Dense

# img224 cls10
def densenet121_img224_cls10(dropout=0.2):
    return Dense(3, 64, 10, 7, 32, 0.5, 4, dropout, 6, 12, 24, 16)

def densenet161_img224_cls10(dropout=0.2):
    return Dense(3, 96, 10, 7, 48, 0.5, 4, dropout, 6, 12, 36, 24)

def densenet169_img224_cls10(dropout=0.2):
    return Dense(3, 64, 10, 7, 32, 0.5, 4, dropout, 6, 12, 32, 32)

def densenet201_img224_cls10(dropout=0.2):
    return Dense(3, 64, 10, 7, 32, 0.5, 4, dropout, 6, 12, 48, 32)

# img224 cls100
def densenet121_img224_cls100(dropout=0.2):
    return Dense(3, 64, 100, 7, 32, 0.5, 4, dropout, 6, 12, 24, 16)

def densenet161_img224_cls100(dropout=0.2):
    return Dense(3, 96, 100, 7, 48, 0.5, 4, dropout, 6, 12, 36, 24)

def densenet169_img224_cls100(dropout=0.2):
    return Dense(3, 64, 100, 7, 32, 0.5, 4, dropout, 6, 12, 32, 32)

def densenet201_img224_cls100(dropout=0.2):
    return Dense(3, 64, 100, 7, 32, 0.5, 4, dropout, 6, 12, 48, 32)

# img32 cls10
def densenet121_img32_cls10(dropout=0.2):
    return Dense(3, 64, 10, 3, 32, 0.5, 4, dropout, 6, 12, 24, 16)

def densenet161_img32_cls10(dropout=0.2):
    return Dense(3, 96, 10, 3, 48, 0.5, 4, dropout, 6, 12, 36, 24)

def densenet169_img32_cls10(dropout=0.2):
    return Dense(3, 64, 10, 3, 32, 0.5, 4, dropout, 6, 12, 32, 32)

def densenet201_img32_cls10(dropout=0.2):
    return Dense(3, 64, 10, 3, 32, 0.5, 4, dropout, 6, 12, 48, 32)

# img32 cls100
def densenet121_img32_cls100(dropout=0.2):
    return Dense(3, 64, 100, 3, 32, 0.5, 4, dropout, 6, 12, 24, 16)

def densenet161_img32_cls100(dropout=0.2):
    return Dense(3, 96, 100, 3, 48, 0.5, 4, dropout, 6, 12, 36, 24)

def densenet169_img32_cls100(dropout=0.2):
    return Dense(3, 64, 100, 3, 32, 0.5, 4, dropout, 6, 12, 32, 32)

def densenet201_img32_cls100(dropout=0.2):
    return Dense(3, 64, 100, 3, 32, 0.5, 4, dropout, 6, 12, 48, 32)

def dense_bc_img32_cls100(l, k, dropout=0.2):
    li = int(((l-1)/3 - 1)//2)
    return Dense(3, 2*k, 100, 3, k, 0.5, 4, dropout, li, li, li)

if __name__ == '__main__':
    pass

