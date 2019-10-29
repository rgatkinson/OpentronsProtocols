::
:: push.cmd
::
:: note: afterwards, robot needs (aparently) to be restarted to pick up new packages
::
ssh -i c:\users\bob\clarence_ssh_key -o StrictHostKeyChecking=no root@clarence.lutzlab.org pip uninstall -y rgatkinson
ssh -i c:\users\bob\clarence_ssh_key -o StrictHostKeyChecking=no root@clarence.lutzlab.org rm -rf /tmp/packaging
scp -i c:\users\bob\clarence_ssh_key -o StrictHostKeyChecking=no -r packaging root@clarence.lutzlab.org:/tmp
ssh -i c:\users\bob\clarence_ssh_key -o StrictHostKeyChecking=no root@clarence.lutzlab.org pip install --no-index /tmp/packaging

:: ssh -i c:\users\bob\clarence_ssh_key -o StrictHostKeyChecking=no root@clarence.lutzlab.org rm -rf /data/packages/usr/local/lib/python3.6/site-packages/rgatkinson-opentrons-enhancements
:: ssh -i c:\users\bob\clarence_ssh_key -o StrictHostKeyChecking=no root@clarence.lutzlab.org rm -rf /data/packages/usr/local/lib/python3.6/site-packages/rgatkinson
:: scp -i c:\users\bob\clarence_ssh_key -o StrictHostKeyChecking=no -r packaging/src/rgatkinson root@clarence.lutzlab.org:/data/packages/usr/local/lib/python3.6/site-packages/rgatkinson
