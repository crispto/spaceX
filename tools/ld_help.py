import subprocess
import re
from os import path
import os

# os.seteuid(some_user_id)
def process(output):
  libs = re.findall(r"/usr.*\.so\.8", output)
  for lib in libs:
    print(lib)
    name = path.basename(lib)
    dir = path.dirname(lib)
    l1 = name
    l2 = name[:-2]
    target = name+".2.1" 
    print(f"{l1}, {l2} -> {target}")
    os.system("rm -rf "+path.join(dir, l1))
    os.system("rm -rf "+path.join(dir, l2))
    os.system("ln -s "+path.join(dir, target)+" "+path.join(dir, l1))
    os.system("ln -s "+path.join(dir, target)+" "+path.join(dir, l2))
    
out = subprocess.run("ldconfig", stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True)
if out.returncode != 0:
    print("return code not 0")
    process(out.stderr)
else:
    print("success")
  



