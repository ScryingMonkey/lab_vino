class LogHelper:
        logLevel = -1
        def log(self,s,level=0):
        if level >= logLevel:
                print(f"{s}")
        def logDict(self,d,name="unnamed",level=0):
        log(f"...printing dictionary {name}:")
        log("{")
        for k in d:
                log(f"{k}:{d[k]}")
        log("}")
        def logfunc(self,f,args):
        print(f">> running {f.__name__}(")
        for k in args:
                log(f"{k}:{args[k]}")
        log(")")