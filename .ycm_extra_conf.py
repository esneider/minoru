import os
import subprocess

def run(cmd):
    return subprocess.Popen(cmd.split(), stdout=subprocess.PIPE).communicate()[0]

FLAGS = ['-std=c++11', '-x c++', '-Wall', '-Wno-unused-variable']

INCLUDES = ['-I.', '-Ilib/cvplot', '-Ilib/eyeLike']

OPENCV = run('pkg-config --libs --cflags opencv').split()

ARMADILLO = ['-Ilib/armadillo/include', '-Llib/armadillo', '-larmadillo']

LOCAL = ['-I/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1']

def FlagsForFile(filename, **kwargs):

    flags = OPENCV + ARMADILLO + FLAGS + INCLUDES + LOCAL
    path  = os.path.dirname(os.path.abspath(__file__))
    final = []

    for flag in flags:
        if flag[:2] in ['-I', '-L']:
            flag = flag[:2] + os.path.join(path, flag[2:])
        final.append(flag)

    return { 'flags': final, 'do_cache': True }
