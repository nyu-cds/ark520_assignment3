from time import sleep
def f(a):
    sleep(a)
    print("doing3 a=%f" % a)
def g():
    print("g3")
    
    
if __name__=='__main__':
    f(0.2)
    print("done")