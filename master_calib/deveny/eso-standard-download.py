import ftplib

import numpy as np

# star = 'G191B2B'.lower()
star = 'aaareadme'
ftp_addr = 'ftp.eso.org'
working_dir = '/pub/stecf/standards/okestan/'
# working_dir = ''
ftp_usr = 'anonymous'


def get_file(ftp, filename):
    try:
        ftp.retrbinary("RETR " + filename, open(filename, 'wb').write)
    except:
        print("Error")


_ftp = ftplib.FTP(ftp_addr)
_ftp.login(ftp_usr)
_ftp.cwd(working_dir)
all_files = []
_ftp.dir(all_files.append)
print(all_files)
f_names = [i.split()[-1] for i in all_files]
print(f_names)
# f_names_star = [i for i in f_names if i.find(star) > -1 and i.endswith('.dat')]
f_names_star = [i for i in f_names if i.find(star) > -1]  # and i.endswith('.dat')]
for f in f_names_star:
    get_file(_ftp, f)
    dat_data = np.loadtxt(f)
    np.save(f.replace('.dat', '.npy'), dat_data)
