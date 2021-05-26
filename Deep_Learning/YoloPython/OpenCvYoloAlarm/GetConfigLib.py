import configparser
import sys, os

def readConfigParam():
    thisdir = os.path.dirname(os.path.abspath(sys.argv[0]))  # directory of script
    cfgfile = os.path.join(thisdir, 'alarmsystem.cfg')
    conf = configparser.RawConfigParser()
    conf.read(cfgfile)
    return conf

def printKeys(conf):
    lkeys = list()
    for i in conf.items('DEFAULT'):
        lkeys.append(i[0])
        #print(i[0])
    print(lkeys)

def printItems(conf):
    for i in conf.items('DEFAULT'):
        print(i[0], " : ", i[1])

def getValue(conf, key):
    return conf.get('DEFAULT', key, fallback=None)

def getValueInt(conf, key):
    val = conf.get('DEFAULT', key, fallback=None)
    if val is not None:
        try:
            ival = int(val)
            return ival
        except:
            return None
    else:
        return None

def getValueFloat(conf, key):
    val = conf.get('DEFAULT', key, fallback=None)
    if val is not None:
        try:
            fval = float(val)
            return fval
        except:
            return None
    else:
        return None

def getValueBoolean(conf, key):
    try:
        val = conf.getboolean('DEFAULT', key, fallback=False)
        return val
    except:
        return False

def getValueBooleanFalse(conf, key):
    try:
        val = conf.getboolean('DEFAULT', key, fallback=True)
        return val
    except:
        return True


