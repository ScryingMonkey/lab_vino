
class LogHelper:
        logLevel = -1

        def log(s,level=0):
                if level >= LogHelper.logLevel:
                        print(f"{s}")

        def logDict(d,name="unnamed",level=0):
                LogHelper.log(f"...printing dictionary {name}:")
                LogHelper.log("{")
                for k in d:
                        LogHelper.log(f"{k}:{d[k]}")
                LogHelper.log("}")

        def logfunc(f,args):
                LogHelper.log(f">> running {f.__name__}(")
                for k in args:
                        LogHelper.log(f">     {k}:{args[k]}")
                LogHelper.log(">> )")