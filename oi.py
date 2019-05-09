import subprocess
import re

import os
a  = os.popen('avprobe videos/test.mp4').readlines()
output = '\n'.join(a)
result = re.search('(\d{3,4})x(\d{3,4})', output)
print(len(output))