logLevel = -1
def log(s,level=0):
    if level >= logLevel:
        print(f"{s}")
def logDict(d,name="unnamed",level=0):
    log(f"...printing dictionary {name}:")
    log("{")
    for k in d:
        log(f"{k}:{d[k]}")
    log("}")